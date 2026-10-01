import unittest

import numpy as np
from PIL import Image

import image_clean


class BackgroundReconstructionTests(unittest.TestCase):
    def test_gray_output_keeps_real_dark_content(self):
        h, w = 180, 180
        y, x = np.mgrid[:h, :w]

        paper = 200 + 30 * np.sin(x / 22.0) + 18 * np.cos(y / 30.0)
        paper = np.clip(paper, 150, 230)
        paper[62:75, 42:65] = 105
        paper[82:94, 92:110] = 127
        paper += np.random.default_rng(0).normal(0, 4, (h, w))
        paper = np.clip(paper, 0, 255)

        rgb = np.dstack([paper, paper, paper]).astype(np.uint8)
        image = Image.fromarray(rgb)

        out, _ = image_clean.clean_page_image(image, user_mode="gray")
        arr = np.array(out.convert("L"))

        content = np.concatenate([
            arr[62:75, 42:65].ravel(),
            arr[82:94, 92:110].ravel(),
        ])
        paper_pixels = arr[arr > 220]

        self.assertGreater(paper_pixels.mean(), 240)
        self.assertLess(content.mean(), paper_pixels.mean() - 15)
        self.assertGreater(content.min(), 100)

    def _page_with_stains(self):
        """Gray-cast page with a wide fold band, a pink ink-bleed patch,
        dark text and a strongly-coloured stamp."""
        h, w = 220, 160
        y, x = np.mgrid[:h, :w]
        rng = np.random.default_rng(1)

        # slow gray cast + mild gradient
        paper = 195 + 15 * np.sin(x / 40.0) + 10 * np.cos(y / 50.0)
        # wide fold band (the "folded paper shape"): broad, 25px tall
        paper[60:85, :] -= 28
        paper = np.clip(paper, 0, 255)

        rgb = np.dstack([paper, paper, paper]).copy()
        # pink ink-bleed: bright, mid-chroma, spatially smooth
        yy, xx = np.ogrid[:h, :w]
        bleed = (yy - 120) ** 2 + (xx - 30) ** 2 <= 12 ** 2
        rgb[bleed, 0] += 45
        rgb[bleed, 1] += 12
        rgb[bleed, 2] += 28
        # dark real text (rods) — must survive
        rgb[18:30, 70:74] = 60
        rgb[18:30, 90:94] = 60
        rgb[110:122, 130:134] = 60
        # a *neutral grey logo*: bright, chroma ~0, thin ink bars that deviate
        # from the paper field — real graphism that must NOT be washed away
        logo = np.zeros((h, w), bool)
        logo[150:166, 20:24] = True
        logo[150:166, 30:34] = True
        logo[150:152, 20:34] = True
        rgb[logo] = 168
        # strong stamp: a realistic *thin-ring* of very saturated red ink
        # (its thickness is below the background-model kernel, so it survives
        # the flat field like real stamp glyphs/rings do) — must survive
        radius = np.sqrt((yy - 40) ** 2 + (xx - 120) ** 2)
        stamp = (radius >= 8.5) & (radius <= 11.5)
        rgb[stamp] = [200, 45, 45]
        rgb += rng.normal(0, 2, rgb.shape)
        return np.clip(rgb, 0, 255).astype(np.uint8), (h, w, y, x, bleed, stamp)

    def test_color_wash_removes_fold_and_bleed_keeps_stamp_and_text(self):
        rgb, (h, w, y, x, bleed, stamp) = self._page_with_stains()
        out, info = image_clean.clean_page_image(
            Image.fromarray(rgb), user_mode="color")
        arr = np.asarray(out.convert("RGB"), dtype=np.int16)
        luma = (arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114)
        chroma = arr.max(axis=2) - arr.min(axis=2)

        self.assertEqual(info["mode"], "color")

        # fold band interior: now clean white, no tint
        fold = np.zeros((h, w), bool)
        fold[65:80, 10:150] = True
        fold &= ~self._rods()
        self.assertGreater(luma[fold].mean(), 248, "fold band still grey")
        self.assertLess(chroma[fold].mean(), 6, "fold band still tinted")

        # ink-bleed centre: neutral white again
        bleed_core = (bleed & (np.abs(x) < 1e9)) | (
            (y - 120) ** 2 + (x - 30) ** 2 <= 5 ** 2)
        self.assertGreater(luma[bleed_core].mean(), 245, "bleed still visible")
        self.assertLess(chroma[bleed_core].mean(), 12, "bleed tint kept")

        # dark text rods keep their ink
        text = np.zeros((h, w), bool)
        for r0, r1, c0, c1 in ((18, 30, 70, 74), (18, 30, 90, 94),
                               (110, 122, 130, 134)):
            text[r0:r1, c0:c1] = True
        self.assertLess(luma[text].mean(), 200, "text was washed away")

        # the stamp's thin ink ring stays strongly coloured, and the paper
        # inside the ring (no ink) is washed clean like the rest of the sheet
        ring = (np.sqrt((y - 40) ** 2 + (x - 120) ** 2) >= 9.0) & (
            np.sqrt((y - 40) ** 2 + (x - 120) ** 2) <= 11.0)
        self.assertGreater(chroma[ring].mean(), 90,
                           "real stamp was washed out")
        inner = np.sqrt((y - 40) ** 2 + (x - 120) ** 2) <= 4.0
        self.assertGreater(luma[inner].mean(), 248,
                           "paper inside the stamp ring kept its tint")

        # the neutral grey logo bars keep their ink: bright neutral graphism
        # is real content, not background, even well above the wash floor
        logo = np.zeros((h, w), bool)
        logo[150:166, 20:24] = True
        logo[150:166, 30:34] = True
        logo[150:152, 20:34] = True
        self.assertLess(luma[logo].mean(), 245,
                        "grey logo bars were washed to white")
        self.assertGreater(luma[logo].mean(), 200, "grey logo bars were crushed")

    @staticmethod
    def _rods():
        m = np.zeros((220, 160), bool)
        for r0, r1, c0, c1 in ((18, 30, 70, 74), (18, 30, 90, 94),
                               (110, 122, 130, 134)):
            m[r0:r1, c0:c1] = True
        return m

    def test_gray_wash_removes_fold_band(self):
        rgb, (h, w, y, x, bleed, stamp) = self._page_with_stains()
        grey = np.asarray(Image.fromarray(rgb).convert("L"))
        out, _ = image_clean.clean_page_image(Image.fromarray(grey), user_mode="gray")
        arr = np.asarray(out.convert("L"), dtype=np.float32)

        fold = np.zeros((h, w), bool)
        fold[65:80, 10:150] = True
        fold &= ~self._rods()
        self.assertGreater(arr[fold].mean(), 245,
                           "gray-mode fold band still grey")

    def test_fold_exposure_map_finds_physical_fold_only(self):
        """A wide illumination fold deviates from its smooth sheet; clean
        paper and thin edges do not."""
        h, w = 200, 200
        y, x = np.mgrid[:h, :w]
        field = 200 + 12 * np.sin(x / 45.0)           # slow shading ramp
        field[80:110, :] -= 20                        # the fold depression
        field = np.clip(field, 0, 255).astype(np.uint8)

        zone = image_clean._fold_exposure_map(field.astype(np.float32))
        self.assertGreater(zone[85:105, 20:180].mean(), 0.4,
                           "fold depression not detected")
        self.assertLess(zone[10:50, 20:180].mean(), 0.05,
                        "clean paper flagged as folded")
        self.assertLess(zone[110:170, :].mean(), 0.05,
                        "sheet gradient flagged as folded")

    def test_fold_ink_suppresses_shadow_keeps_deep_and_coloured_ink(self):
        """Inside a folded zone, smooth neutral twilight-band dark (150..wash
        floor) is crease shadow and is suppressed; deep text and coloured ink
        survive; outside the zone nothing is touched."""
        h, w = 200, 200
        fold = np.zeros((h, w), bool)
        fold[70:105, :] = True

        l2 = np.full((h, w), 255.0, dtype=np.float32)
        # deep text rods — one inside the fold, one outside (below the
        # ink floor: real ink is deeper than the twilight band)
        l2[82:92, 40:44] = 90.0
        l2[150:160, 40:44] = 90.0
        # smooth twilight crease shadow inside the fold -> must go
        l2[75:83, 120:132] = 185.0
        # same twilight dark OUTSIDE the fold -> must be kept
        l2[120:128, 150:160] = 185.0

        chroma = np.zeros((h, w), dtype=np.float32)
        # strongly coloured ink stroke crossing the fold (stamp / felt-pen)
        chroma[74:80, 60:74] = 110.0

        grad = np.zeros_like(l2)
        target = image_clean._fold_ink(l2, fold, grad, chroma)

        self.assertGreater(target[75:83, 120:132].mean(), 0.2,
                           "crease shadow inside fold not suppressed")
        self.assertLess(target[120:128, 150:160].sum(), 5,
                        "twilight dark outside the fold zone was suppressed")
        self.assertLess(target[82:92, 40:44].sum(), 5,
                        "deep text inside the fold was suppressed")
        self.assertLess(target[150:160, 40:44].sum(), 5,
                        "deep text outside the fold was suppressed")
        self.assertLess(target[74:80, 60:74].sum(), 5,
                        "coloured ink inside the fold was suppressed")



    def test_corner_fold_mask_isolated_to_corner(self):
        h, w = 300, 300
        y, x = np.mgrid[:h, :w]
        base = np.full((h, w), 235.0, dtype=np.float32)
        # Smooth dark dog-ear-like wedge touching the top-left corner.
        base -= 42.0 * np.exp(-((x + y) / 34.0) ** 2)
        base += np.random.default_rng(3).normal(0, 0.6, (h, w))
        base = np.clip(base, 0, 255)
        bg = np.full_like(base, 235.0)
        mask = image_clean._corner_fold_mask(base, bg)
        self.assertGreater(mask[:55, :55].mean(), 0.05,
                           "corner fold was not detected")
        self.assertLess(mask[120:, 120:].mean(), 0.005,
                        "corner detector leaked into page body")

    def test_grayscale_band_mask_removes_long_shallow_streak_not_text(self):
        h, w = 260, 360
        base = np.full((h, w), 238.0, dtype=np.float32)
        # Scanner/toner band: long, shallow, smooth grayscale defect.
        base[115:119, 20:340] -= 18.0
        base += np.random.default_rng(4).normal(0, 0.5, (h, w))
        # Genuine dark text-like strokes must not be classified as banding.
        base[70:105, 90:96] = 70.0
        base[70:105, 120:126] = 70.0
        base[150:185, 210:216] = 70.0
        base = np.clip(base, 0, 255)
        bg = np.full_like(base, 238.0)
        mask = image_clean._grayscale_band_mask(base, bg)
        self.assertGreater(mask[115:119, 40:330].mean(), 0.25,
                           "long grayscale band was not detected")
        self.assertLess(mask[65:110, 80:135].mean(), 0.03,
                        "dark text was classified as banding")

class CamScannerReconstructionTests(unittest.TestCase):
    def test_document_mode_rebuilds_paper_and_keeps_dark_text(self):
        h, w = 260, 360
        y, x = np.mgrid[:h, :w]
        rng = np.random.default_rng(9)
        paper = 188 + 28 * np.sin(x / 43.0) + 18 * np.cos(y / 37.0)
        paper[112:132, :] -= 24
        paper += rng.normal(0, 2.0, (h, w))
        paper[55:105, 85:91] = 55
        paper[55:105, 125:131] = 65
        paper[175:188, 180:250] = 70
        rgb = np.dstack([paper, paper, paper]).clip(0, 255).astype(np.uint8)

        out, info = image_clean.clean_page_image(
            Image.fromarray(rgb), user_mode="document"
        )
        arr = np.asarray(out.convert("L"), dtype=np.float32)
        self.assertEqual(info["mode"], "document")

        background = np.ones((h, w), dtype=bool)
        background[55:105, 85:91] = False
        background[55:105, 125:131] = False
        background[175:188, 180:250] = False
        self.assertGreater(arr[background].mean(), 247.0)
        self.assertLess(arr[55:105, 85:91].mean(), 180.0)
        self.assertLess(arr[55:105, 125:131].mean(), 190.0)


if __name__ == "__main__":
    unittest.main()
