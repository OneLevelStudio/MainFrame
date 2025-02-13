from lib.PaddleOCR.PaddleOCR import Process_PaddleOCR as PaddleOCR
from lib.VietOCR.VietOCR import Process_VietOCR as VietOCR
from PIL import Image

def create_final_ocr_text(ocr_bboxes, ocr_texts, same_line_max_bbox_y_diff=10, new_line_char="\n", same_line_char=" | "):

    e01s = [e[0][1] for e in ocr_bboxes]
    y_diff = [e01s[i+1] - e01s[i] for i in range(len(e01s)-1)]
    y_diff_binary = [e > same_line_max_bbox_y_diff for e in y_diff]

    groups_idxs = []
    for i in range(len(y_diff_binary)):
        if i == 0:
            g_tmp = [0]
        if y_diff_binary[i] == True:
            groups_idxs.append(g_tmp)
            g_tmp = []
        g_tmp.append(i+1)
        if (i==len(y_diff_binary)-1):
            groups_idxs.append(g_tmp)

    groups_xsorted_idxs = []
    for e in groups_idxs:
        tmp = [ocr_bboxes[u][0][0] for u in e]
        paired = list(zip(tmp, e))
        paired.sort()
        _sorted_tmp, sorted_e = zip(*paired)
        groups_xsorted_idxs.append(sorted_e)

    groups_txts = []
    for e in groups_xsorted_idxs:
        tmp = []
        for u in e:
            tmp.append(ocr_texts[u])
        groups_txts.append(tmp)

    final_ocr_text = new_line_char.join([same_line_char.join(e) for e in groups_txts])

    return final_ocr_text

def Process_VIOCR(img_path, bbox_padding=2, same_line_max_bbox_y_diff=10, new_line_char="\n", same_line_char=" | ", print_debug=True):
    img_og = Image.open(img_path)

    # Text Detection (PaddleOCR)
    if print_debug:
        print("> Processing: Text Detection...")
    ocr_bboxes = PaddleOCR(img_path, padding=bbox_padding, debug_dot=print_debug)

    # Text Recognition (VietOCR)
    if print_debug:
        print("\n> Processing: Text Recognition...")
    ocr_crop_bboxes = [img_og.crop((bb[0][0], bb[0][1], bb[1][0], bb[1][1])) for bb in ocr_bboxes]
    ocr_texts = [VietOCR(img_crop, debug_dot=print_debug) for img_crop in ocr_crop_bboxes]

    # Sort by bbox y
    if print_debug:
        print("\n> Processing: Sorting...")
    ocr_paired = list(zip(ocr_bboxes, ocr_texts))
    ocr_paired = sorted(ocr_paired, key=lambda x: x[0][0][1])
    ocr_bboxes, ocr_texts = zip(*ocr_paired)

    # Final text
    if print_debug:
        print("> Processing: Final text...")
    final_ocr_text = create_final_ocr_text(
        ocr_bboxes, ocr_texts, 
        same_line_max_bbox_y_diff = same_line_max_bbox_y_diff, 
        new_line_char = new_line_char, 
        same_line_char = same_line_char
    )
    if print_debug:
        print("==================================================")
        print(final_ocr_text)

    # Return
    return final_ocr_text