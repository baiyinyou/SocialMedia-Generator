from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
# imagegen.py
from PIL import Image, ImageDraw, ImageFont

def render_cover_image(title_text: str, subtitle_text: str, font_path: str = None):

    W, H = 1024, 576
    img = Image.new("RGB", (W, H), (22, 28, 36))
    draw = ImageDraw.Draw(img)

    # 背景渐变
    for y in range(H):
        c = int(22 + (y / H) * 60)
        draw.line([(0, y), (W, y)], fill=(c, c, c))

    # 字体加载
    try:
        if font_path and os.path.exists(font_path):
            title_font = ImageFont.truetype(font_path, 58)
            sub_font = ImageFont.truetype(font_path, 32)
        else:
            title_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    # 文字绘制参数
    margin = 60
    max_width = W - 2 * margin

    # 自动换行函数
    def wrap(text, font):
        words = text.split()
        lines, line = [], ""
        for w in words:
            if draw.textlength(line + " " + w, font=font) < max_width:
                line += " " + w
            else:
                lines.append(line.strip())
                line = w
        if line:
            lines.append(line.strip())
        return lines

    title_lines = wrap(title_text, title_font)
    subtitle_lines = wrap(subtitle_text, sub_font)

    # 绘制标题和副标题
    y = H // 3
    x = margin
    for line in title_lines:
        draw.text((x, y), line, fill=(240, 250, 255), font=title_font)
        y += title_font.size + 10
    for line in subtitle_lines:
        draw.text((x, y), line, fill=(200, 210, 220), font=sub_font)
        y += sub_font.size + 5

    # 添加右下角角标
    tag = "LinkedIn Insight"
    tbox = draw.textbbox((0, 0), tag, font=sub_font)
    tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
    draw.rectangle(
        [(W - tw - 40, H - th - 40), (W - 40, H - 20)],
        fill=(30, 150, 255), outline=None
    )
    draw.text((W - tw - 30, H - th - 38), tag, font=sub_font, fill=(255, 255, 255))

    return img
