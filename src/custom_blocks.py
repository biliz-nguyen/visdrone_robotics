import torch
import torch.nn as nn
import torch.nn.functional as F


class SPRDown(nn.Module):
    """Selective Phase Reassembly downsampling for YOLOEdge27.

    The module decomposes a feature map into the four stride-2 polyphase
    components, estimates lightweight channel-wise phase weights from the
    input itself, reassembles the phases without concatenating to 4C, then
    applies depthwise spatial mixing and pointwise channel projection.

    Design goals:
      - preserve sampling-phase evidence before resolution reduction;
      - avoid the 4C expansion of PixelUnshuffle-style downsampling;
      - keep the deploy path composed of common tensor/Conv/BN/SiLU ops.
    """

    default_act = nn.SiLU()

    def __init__(
        self,
        c1,
        c2,
        k=3,
        s=2,
        p=None,
        g=1,
        d=1,
        act=True,
        temperature=1.0,
    ):
        super().__init__()

        if int(s) != 2:
            raise ValueError(f"SPRDown is a stride-2 block, got s={s}")
        if int(k) != 3:
            raise ValueError(f"SPRDown v1 expects k=3, got k={k}")
        if int(g) != 1:
            raise ValueError(f"SPRDown v1 expects g=1, got g={g}")
        if int(d) != 1:
            raise ValueError(f"SPRDown v1 expects d=1, got d={d}")
        if float(temperature) <= 0:
            raise ValueError("SPRDown temperature must be > 0")

        self.c1 = int(c1)
        self.c2 = int(c2)
        self.k = int(k)
        self.s = int(s)
        self.temperature = float(temperature)

        # Two tiny per-channel parameter sets control the response of the
        # four phases. 8*C parameters in total, independent of H/W.
        self.phase_scale = nn.Parameter(
            torch.ones(1, self.c1, 4)
        )
        self.phase_bias = nn.Parameter(
            torch.zeros(1, self.c1, 4)
        )

        activation = (
            nn.SiLU(inplace=True)
            if act is True
            else (
                act
                if isinstance(act, nn.Module)
                else nn.Identity()
            )
        )

        self.spatial_mix = nn.Sequential(
            nn.Conv2d(
                self.c1,
                self.c1,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.c1,
                bias=False,
            ),
            nn.BatchNorm2d(self.c1),
            nn.SiLU(inplace=True)
            if act is True
            else nn.Identity(),
        )

        self.channel_mix = nn.Sequential(
            nn.Conv2d(
                self.c1,
                self.c2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.c2),
            activation,
        )

    @staticmethod
    def _pad_to_even(x):
        """Pad only the right/bottom edge when an odd spatial size appears."""
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="replicate",
            )
        return x

    @staticmethod
    def _polyphase_split(x):
        """Return B,C,4,H/2,W/2 polyphase components."""
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        return torch.stack(
            (x00, x01, x10, x11),
            dim=2,
        )

    def phase_weights(self, phases):
        """Estimate input-conditioned channel-wise weights over 4 phases."""
        # Absolute activation is used as a sign-agnostic phase descriptor.
        descriptor = phases.abs().mean(dim=(-1, -2))
        scores = (
            descriptor * self.phase_scale
            + self.phase_bias
        ) / self.temperature
        return torch.softmax(scores, dim=2)

    def forward(self, x):
        if x.shape[1] != self.c1:
            raise RuntimeError(
                f"SPRDown expected {self.c1} channels, got {x.shape[1]}"
            )

        x = self._pad_to_even(x)
        phases = self._polyphase_split(x)
        weights = self.phase_weights(phases)

        # Reassembly compresses four sampling phases back to C channels.
        y = (
            phases
            * weights.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=2)

        y = self.spatial_mix(y)
        y = self.channel_mix(y)
        return y

    def extra_repr(self):
        return (
            f"c1={self.c1}, c2={self.c2}, stride=2, "
            f"temperature={self.temperature}"
        )


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
