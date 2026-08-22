import torch
import torch.nn as nn


class AConv(nn.Module):
    """25% detail / 25% context / 50% preserve."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        d=1,
        act=True,
    ):
        super().__init__()

        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s

        activation = (
            nn.SiLU(inplace=True)
            if act is True
            else (
                act
                if isinstance(act, nn.Module)
                else nn.Identity()
            )
        )

        if k != 3 or g != 1:
            pad = (
                (d * (k - 1) + 1) // 2
                if p is None
                else p
            )
            self.mode = "standard"
            self.standard = nn.Sequential(
                nn.Conv2d(
                    c1,
                    c2,
                    k,
                    s,
                    pad,
                    dilation=d,
                    groups=g,
                    bias=False,
                ),
                nn.BatchNorm2d(c2),
                activation,
            )
            return

        self.mode = "partial"

        self.c_detail = max(1, c1 // 4)
        self.c_context = max(1, c1 // 4)
        self.c_preserve = (
            c1
            - self.c_detail
            - self.c_context
        )

        if self.c_preserve <= 0:
            raise ValueError(
                f"Invalid AConv split for c1={c1}"
            )

        self.detail = nn.Sequential(
            nn.Conv2d(
                self.c_detail,
                self.c_detail,
                3,
                s,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(self.c_detail),
            nn.SiLU(inplace=True)
            if act
            else nn.Identity(),
        )

        self.context = nn.Sequential(
            nn.Conv2d(
                self.c_context,
                self.c_context,
                5,
                s,
                2,
                groups=self.c_context,
                bias=False,
            ),
            nn.BatchNorm2d(self.c_context),
            nn.SiLU(inplace=True)
            if act
            else nn.Identity(),
        )

        self.preserve = (
            nn.Identity()
            if s == 1
            else nn.Sequential(
                nn.Conv2d(
                    self.c_preserve,
                    self.c_preserve,
                    3,
                    s,
                    1,
                    groups=self.c_preserve,
                    bias=False,
                ),
                nn.BatchNorm2d(self.c_preserve),
                nn.SiLU(inplace=True)
                if act
                else nn.Identity(),
            )
        )

        self.mix = nn.Sequential(
            nn.Conv2d(
                c1,
                c2,
                1,
                1,
                0,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            activation,
        )

    def forward(self, x):
        if self.mode == "standard":
            return self.standard(x)

        xd, xc, xp = torch.split(
            x,
            [
                self.c_detail,
                self.c_context,
                self.c_preserve,
            ],
            dim=1,
        )

        yd = self.detail(xd)
        yc = self.context(xc)
        yp = self.preserve(xp)

        if (
            yd.shape[-2:] != yc.shape[-2:]
            or yd.shape[-2:] != yp.shape[-2:]
        ):
            raise RuntimeError(
                "AConv branch mismatch: "
                f"{yd.shape}, {yc.shape}, {yp.shape}"
            )

        return self.mix(
            torch.cat([yd, yc, yp], dim=1)
        )


class ECA(nn.Module):
    def __init__(self, c1, c2, k_size=3):
        super().__init__()

        if c1 != c2:
            raise ValueError(
                f"ECA requires c1==c2, got {c1}->{c2}"
            )

        if k_size % 2 == 0:
            raise ValueError("ECA kernel must be odd")

        self.c1 = c1
        self.c2 = c2

        self.conv = nn.Conv2d(
            1,
            1,
            kernel_size=(k_size, 1),
            padding=(k_size // 2, 0),
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3), keepdim=True)
        y = y.permute(0, 2, 1, 3)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1, 3)
        return x * y


class CoordinateGate(nn.Module):
    def __init__(
        self,
        c1,
        reduction=32,
        min_channels=8,
    ):
        super().__init__()

        hidden = max(
            min_channels,
            c1 // reduction,
        )

        self.conv1 = nn.Conv2d(
            c1,
            hidden,
            1,
            1,
            0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(
            hidden,
            c1,
            1,
            1,
            0,
            bias=True,
        )
        self.conv_w = nn.Conv2d(
            hidden,
            c1,
            1,
            1,
            0,
            bias=True,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]

        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True)
        x_w = x_w.permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y_h, y_w = torch.split(
            y,
            [h, w],
            dim=2,
        )

        y_w = y_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))

        return a_h, a_w


class CoordAtt(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        reduction=32,
        min_channels=8,
    ):
        super().__init__()

        if c1 != c2:
            raise ValueError(
                f"CoordAtt requires c1==c2, got {c1}->{c2}"
            )

        self.c1 = c1
        self.c2 = c2

        self.gate = CoordinateGate(
            c1,
            reduction=reduction,
            min_channels=min_channels,
        )

    def forward(self, x):
        a_h, a_w = self.gate(x)
        return x * a_h * a_w


class ResidualLiteCA(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        reduction=32,
        min_channels=8,
        alpha_init=0.10,
    ):
        super().__init__()

        if c1 != c2:
            raise ValueError(
                "ResidualLiteCA requires "
                f"c1==c2, got {c1}->{c2}"
            )

        self.c1 = c1
        self.c2 = c2

        self.gate = CoordinateGate(
            c1,
            reduction=reduction,
            min_channels=min_channels,
        )

        self.alpha = nn.Parameter(
            torch.tensor(float(alpha_init))
        )

    def forward(self, x):
        a_h, a_w = self.gate(x)
        refined = x * a_h * a_w
        return x + self.alpha * refined
