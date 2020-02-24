import numpy as np
import torch
import cv2


def labels_to_imgs(img_shape, resolve_texts):

    bs, ch, h, w = img_shape
    data = np.zeros((bs, ch, h, w), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 70 * (w * h) / (1000 * 1000) # Would work best for almost square images
    fontColor = (255, 255, 255)
    lineType = 2
    thickness = 1

    for i in range(img_shape[0]):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        resolve_text = resolve_texts[i]
        textsize = cv2.getTextSize(resolve_text, font, fontScale, thickness)[0]

        text_w, text_h = textsize
        coord_x = w // 2 - text_w // 2
        coord_y = h // 2

        cv2.putText(img, resolve_text,
                    (coord_x, coord_y),
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        img = np.transpose(img, (2, 0, 1))
        data[i] = img

    return torch.from_numpy(data).to(torch.float32)