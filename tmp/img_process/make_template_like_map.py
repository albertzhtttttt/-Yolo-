from __future__ import annotations

"""
make_template_like_map.py - 仿 template.png 的地图复合版式生成脚本
路径：tmp/img_process/make_template_like_map.py

功能：
    读取同目录下的 left.png、中国底图.png、深圳市底图.png，生成类似 template.png 的
    左侧主图 + 右侧定位 inset + 图例/色带的论文地图版式图。
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "template_like_result.png"
OUT_PDF = ROOT / "template_like_result.pdf"

CANVAS_W = 900
CANVAS_H = 674
SCALE = 3


def load_font(size: int) -> ImageFont.ImageFont:
    """优先加载常见中文字体，失败时回退到 PIL 默认字体。"""
    candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def crop_non_background(image: Image.Image, bg_color: tuple[int, int, int] | None = None, threshold: int = 12) -> Image.Image:
    """裁掉纯黑或纯白背景的大面积留白，保留地图主体。"""
    image = image.convert("RGBA")
    rgb = image.convert("RGB")

    if bg_color is None:
        bg_color = rgb.getpixel((0, 0))

    pixels = rgb.load()
    width, height = rgb.size
    xs = []
    ys = []

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if abs(r - bg_color[0]) > threshold or abs(g - bg_color[1]) > threshold or abs(b - bg_color[2]) > threshold:
                xs.append(x)
                ys.append(y)

    if not xs:
        return image

    pad = max(8, min(width, height) // 80)
    left = max(0, min(xs) - pad)
    upper = max(0, min(ys) - pad)
    right = min(width, max(xs) + pad)
    lower = min(height, max(ys) + pad)
    return image.crop((left, upper, right, lower))


def fit_image(image: Image.Image, box_size: tuple[int, int], fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """等比例缩放图片并居中放入指定尺寸画布。"""
    image = image.convert("RGBA")
    fitted = ImageOps.contain(image, box_size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", box_size, fill + (255,))
    x = (box_size[0] - fitted.width) // 2
    y = (box_size[1] - fitted.height) // 2
    canvas.alpha_composite(fitted, (x, y))
    return canvas


def paste_panel(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    draw: ImageDraw.ImageDraw,
    border: tuple[int, int, int] = (40, 40, 40),
) -> None:
    """把图片贴入指定框，并绘制细边框。"""
    x, y, w, h = box
    canvas.alpha_composite(image, (x, y))
    draw.rectangle((x, y, x + w, y + h), outline=border, width=2)


def add_image_border(image: Image.Image, border_width: int = 2, border_color: tuple[int, int, int] = (40, 40, 40)) -> Image.Image:
    """直接给图片本体加边框，不再先放入额外底板。"""
    image = image.convert("RGBA")
    return ImageOps.expand(image, border=border_width, fill=border_color)


def draw_legend(draw: ImageDraw.ImageDraw, origin: tuple[int, int], small_font: ImageFont.ImageFont) -> None:
    """仅保留一个图例项，并按用户要求改成 Area of data acquisition。"""
    x, y = origin
    draw.rectangle((x, y, x + 28, y + 28), fill=(40, 90, 185), outline=(20, 20, 20), width=1)
    draw.text((x + 42, y + 2), "Area of data\nacquisition", fill=(20, 20, 20), font=small_font, spacing=2)


def trim_canvas_whitespace(image: Image.Image, margin: int = 10, threshold: int = 8) -> Image.Image:
    """按最终成图内容边界裁掉外层白边，只保留一圈很薄的统一留白。"""
    image = image.convert("RGBA")
    rgb = image.convert("RGB")
    pixels = rgb.load()
    width, height = rgb.size

    xs = []
    ys = []
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if abs(r - 255) > threshold or abs(g - 255) > threshold or abs(b - 255) > threshold:
                xs.append(x)
                ys.append(y)

    if not xs:
        return image

    left = max(0, min(xs) - margin)
    top = max(0, min(ys) - margin)
    right = min(width, max(xs) + margin + 1)
    bottom = min(height, max(ys) + margin + 1)
    return image.crop((left, top, right, bottom))


def main() -> None:
    """生成仿模板的地图复合图。"""
    main_img = Image.open(ROOT / "left.png")
    china_img = Image.open(ROOT / "中国底图.png")
    shenzhen_img = Image.open(ROOT / "深圳市底图.png")

    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    small_font = load_font(12)

    # 直接按参考图 image.png 的紧凑程度排版。
    main_max = (640, 520)
    main_crop = crop_non_background(main_img, bg_color=(255, 255, 255), threshold=18)
    main_resized = ImageOps.contain(main_crop, main_max, method=Image.Resampling.LANCZOS)
    main_panel = add_image_border(main_resized, border_width=2)
    main_x = 20
    main_y = 15
    canvas.alpha_composite(main_panel, (main_x, main_y))

    max_inset = (135, 135)
    right_col_x = main_x + main_panel.width + 18

    china_crop = crop_non_background(china_img, bg_color=(0, 0, 0), threshold=18)
    china_panel = add_image_border(ImageOps.contain(china_crop, max_inset, method=Image.Resampling.LANCZOS), border_width=1)
    china_x = right_col_x + (max_inset[0] - china_panel.width) // 2
    china_y = 18
    canvas.alpha_composite(china_panel, (china_x, china_y))

    shenzhen_crop = crop_non_background(shenzhen_img, bg_color=(255, 255, 255), threshold=18)
    shenzhen_panel = add_image_border(ImageOps.contain(shenzhen_crop, max_inset, method=Image.Resampling.LANCZOS), border_width=1)
    shenzhen_x = right_col_x + (max_inset[0] - shenzhen_panel.width) // 2
    shenzhen_y = 178
    canvas.alpha_composite(shenzhen_panel, (shenzhen_x, shenzhen_y))

    # 连接线同步缩短，避免视觉松散。
    main_anchor_x = main_x + main_panel.width
    draw.line((main_anchor_x, 135, china_x, china_y + china_panel.height // 2), fill=(110, 110, 110), width=1)
    draw.line((main_anchor_x, 320, shenzhen_x, shenzhen_y + shenzhen_panel.height // 2), fill=(110, 110, 110), width=1)

    draw_legend(draw, (right_col_x + 8, 350), small_font)

    # 最终成图再统一裁一遍外层白边，目标是达到参考图那种紧凑留白，而不是简单整体缩放。
    final_image = trim_canvas_whitespace(canvas, margin=10, threshold=8)
    final_image.save(OUT_PATH)
    final_image.convert("RGB").save(OUT_PDF)
    print(OUT_PATH)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
