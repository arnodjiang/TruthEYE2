
import os
import cv2
import json
import numpy as np

TARGET_DIR = "/public/data_share/model_hub/lab10_data/trutheye2/datas/sft/miml_part2"


def extract_bboxes_from_mask(mask):
    """给定二值 mask（numpy），返回多个 bbox: [(x1,y1,x2,y2), ...]"""
    h, w = mask.shape[:2]

    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w_box), int(y + h_box)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        bboxes.append([x1, y1, x2, y2])

    return bboxes


def save_case(original_path, mask_path, save_root):
    """
    保存 original, mask, bbox visualization, crops，并生成 json
    """
    base = os.path.splitext(os.path.basename(original_path))[0]
    out_dir = os.path.join(save_root, base)
    os.makedirs(out_dir, exist_ok=True)

    # ========== 读取图片 ==========
    img = cv2.imread(original_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"[Skip] Cannot read {original_path} or {mask_path}")
        return

    # ========== 生成 bbox ==========
    bboxes = extract_bboxes_from_mask(mask)

    # ========== 绘制 bbox 可视化图 ==========
    viz = img.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # ========== 保存基础图片 ==========
    raw_img_path = os.path.join(out_dir, "original.jpg")
    mask_img_path = os.path.join(out_dir, "mask.jpg")
    bbox_img_path = os.path.join(out_dir, "viz.jpg")

    cv2.imwrite(raw_img_path, img)
    cv2.imwrite(mask_img_path, mask)
    cv2.imwrite(bbox_img_path, viz)

    # ========== 裁剪 crop ==========
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    crop_entries = []

    for idx, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
        crop = img[y1:y2, x1:x2]

        crop_path = os.path.join(crops_dir, f"{idx}.jpg")
        cv2.imwrite(crop_path, crop)

        crop_entries.append({
            "bbox": [x1, y1, x2, y2],
            "crop_path": crop_path
        })

    # ========== 保存 JSON ==========
    json_path = os.path.join(out_dir, "metadata.json")
    json_data = {
        "crop_img_path": crop_entries,
        "raw_img_path": raw_img_path,
        "bbox_img_path": bbox_img_path,
        "mask_img_path": mask_img_path
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return json_data


def main(original_dir, mask_dir):
    os.makedirs(TARGET_DIR, exist_ok=True)

    files = sorted(os.listdir(original_dir))
    count = 0

    to_datas = []
    for f in files:
        # if count >= 5:
        #     break

        original_path = os.path.join(original_dir, f)
        if not os.path.isfile(original_path):
            continue

        base = os.path.splitext(f)[0]
        mask_path = os.path.join(mask_dir, base + ".png")  # 若需要改为 .jpg 自行修改

        if not os.path.exists(mask_path):
            print(f"[Skip] mask not found for {f}")
            continue

        try:
            to_itm_data = save_case(original_path, mask_path, TARGET_DIR)
        except Exception as e:
            print(f"[Error] processing {f} failed: {e}")
            continue
        to_datas.append(to_itm_data)
        count += 1
    with open(os.path.join(TARGET_DIR, "miml_visual_data.json"), "w", encoding="utf-8") as f:
        json.dump(to_datas, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Total processed = {count}")


# 示例调用
main("/public/data_share/model_hub/lab10_data/trutheye2/datas/MIML/MIML_Part2/imgs/", "/public/data_share/model_hub/lab10_data/trutheye2/datas/MIML/MIML_Part2/masks/")
