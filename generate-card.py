# pip install pillow requests
from io import BytesIO
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps
import math
import argparse

PORTRAIT_DEFAULT = "https://art.hearthstonejson.com/v1/256x/BGS_104.webp"
BORDER_DEFAULT   = "https://static.hsreplay.net/static/webpack/assets/images/minions/border.5715229ec663177ce8f9.png"
TIER_DEFAULT     = "https://static.hsreplay.net/static/webpack/assets/images/battlegrounds/tavern-tiers/5.7c89576b8a39923564e8.png"

CARD_W, CARD_H = 140, 168

# Ellipse mask sizes (pixels) — tune these to match your frame cutout precisely
ELLIPSE_INSET_LEFT   = math.ceil(0.22 * CARD_W)
ELLIPSE_INSET_RIGHT  = math.ceil(0.22 * CARD_W)
ELLIPSE_INSET_TOP    = math.ceil(0.12 * CARD_H)
ELLIPSE_INSET_BOTTOM = math.ceil(0.18 * CARD_H)

HEALTH_POS_X = math.ceil(0.7 * CARD_W)
HEALTH_POS_Y = math.ceil(0.68 * CARD_H)

ATK_POS_X = math.ceil(0.3 * CARD_W)
ATK_POS_Y = math.ceil(0.68 * CARD_H)

TIER_POS_X = math.ceil(0.5 * CARD_W)
TIER_POS_Y = math.ceil(0.12 * CARD_H)

def _fetch_image(url) -> Image.Image:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content))
    return img.convert("RGBA")

def _load_font(sz: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=sz)
    except Exception:
        return ImageFont.load_default()

def render_minion(
    atk: str = "4",
    hp: str = "4",
    portrait_url: str = PORTRAIT_DEFAULT,
    border_url: str = BORDER_DEFAULT,
    tier_url: str = TIER_DEFAULT,
    out_path: str = "minion.png",
):
    full_border = _fetch_image(border_url)

    # Transparent canvas
    canvas = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # ---- ELLIPTICAL CLIP AREA ----
    # Define ellipse bbox inside the card
    el_left   = ELLIPSE_INSET_LEFT
    el_top    = ELLIPSE_INSET_TOP
    el_right  = CARD_W - ELLIPSE_INSET_RIGHT
    el_bottom = CARD_H - ELLIPSE_INSET_BOTTOM
    ellipse_box = (el_left, el_top, el_right, el_bottom)
    ellipse_w = el_right - el_left
    ellipse_h = el_bottom - el_top

    # Make a full-size mask with an ellipse cutout
    ellipse_mask = Image.new("L", (CARD_W, CARD_H), 0)
    ImageDraw.Draw(ellipse_mask).ellipse(ellipse_box, fill=255)

    # ---- PORTRAIT INTO OVAL ----
    portrait_raw = _fetch_image(portrait_url)

    # Fit/crop the portrait to the ellipse rectangle, preserving aspect ratio
    # (like background-size: cover).
    portrait_fit = ImageOps.fit(portrait_raw, (ellipse_w, ellipse_h), method=Image.LANCZOS, centering=(0.5, 0.5))

    # Place the fitted portrait onto a transparent layer at the ellipse position
    portrait_layer = Image.new("RGBA", (CARD_W, CARD_H), (0, 0, 0, 0))
    portrait_layer.paste(portrait_fit, (el_left, el_top))

    # Apply mask so only the ellipse area remains visible
    portrait_layer.putalpha(ellipse_mask)

    # Composite portrait layer onto the canvas
    canvas.alpha_composite(portrait_layer)

    # ---- BORDER OVERLAY ----
    border = full_border.resize((CARD_W, CARD_H), Image.LANCZOS)
    canvas.alpha_composite(border)

    canvas.save(out_path, "PNG")

    # ---- STATS ----
    txt_color = (237, 237, 237, 255)
    stroke_color = (0, 0, 0, 200)
    stroke_w = 1
    font = _load_font(15)

    # Attack (bottom-left)
    atk_bbox = draw.textbbox((0, 0), atk, font=font)
    atk_w, atk_h = atk_bbox[2] - atk_bbox[0], atk_bbox[3] - atk_bbox[1]

    # Center the text
    atk_pos_x = ATK_POS_X - (atk_w / 2)
    atk_pos_y = ATK_POS_Y - (atk_h / 2)

    draw.text((atk_pos_x, atk_pos_y), atk, font=font, fill=txt_color,
              stroke_width=stroke_w, stroke_fill=stroke_color)

    # Health (bottom-right)
    hp_bbox = draw.textbbox((0, 0), hp, font=font)
    hp_w, hp_h = hp_bbox[2] - hp_bbox[0], hp_bbox[3] - hp_bbox[1]

    # Center the text
    hp_pos_x = HEALTH_POS_X - (hp_w / 2)
    hp_pos_y = HEALTH_POS_Y - (hp_h / 2)

    draw.text((hp_pos_x, hp_pos_y), hp, font=font, fill=txt_color,
              stroke_width=stroke_w, stroke_fill=stroke_color)

    # ---- TIER ICON ----
    tier = _fetch_image(tier_url)

    TIER_SIZE_W = math.ceil(0.25 * CARD_W)
    TIER_SIZE_H = math.ceil(0.22 * CARD_H)

    tier = tier.resize((TIER_SIZE_W, TIER_SIZE_H), Image.LANCZOS)

    # Center the tier img
    tier_pos_x = math.ceil(TIER_POS_X - (TIER_SIZE_W / 2))
    tier_pos_y = math.ceil(TIER_POS_Y - (TIER_SIZE_H / 2))

    print(tier_pos_x)
    print(TIER_POS_X)

    canvas.alpha_composite(tier, (tier_pos_x, tier_pos_y))

    # ---- SAVE ----
    canvas.save(out_path, "PNG")
    print(f"✅ Rendered {out_path}")

# ---- CLI ----
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a Battlegrounds-style minion card."
    )
    p.add_argument("--output", "-o", default="minion.png", help="Output file path (PNG).")
    p.add_argument("--portrait", "-p", default=PORTRAIT_DEFAULT, help="Portrait URL or local path.")
    p.add_argument("--border", "-b", default=BORDER_DEFAULT, help="Border URL or local path.")
    p.add_argument("--tier", "-t", default=TIER_DEFAULT, help="Tier icon URL or local path.")
    p.add_argument("--atk", default="4", help="Attack value text.")
    p.add_argument("--hp", default="4", help="Health value text.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    render_minion(
        atk=args.atk,
        hp=args.hp,
        portrait_url=args.portrait,
        border_url=args.border,
        tier_url=args.tier,
        out_path=args.output,
    )


if __name__ == "__main__":
    main()
