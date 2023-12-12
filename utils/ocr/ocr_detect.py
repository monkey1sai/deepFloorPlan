import torch
import cv2
import easyocr

def orc_parse(image_url):
    im = cv2.imread(image_url)
    reader = easyocr.Reader(['ch_tra','en'])
    result = reader.readtext(im)
    print(result)



# def ocr_parse(images):
#     for i, image in enumerate(images):
#         image.save(os.path.join(abspath, f'check/check_{i}.jpg'), 'JPEG')
#         time.sleep(0.1)
#         #import easyocr
#         #reader = easyocr.Reader(['ch_sim','en'])
#         #result = reader.readtext('93351929_93351931.png')
#         #for res in result:
#         #    print(res)
#         start_time = time.time()
#         for i in range(10):
#             reader = easyocr.Reader(['ch_tra','en'])
#             img = cv2.imread(os.path.join(abspath, f'check/check_{i}.jpg'))
#             result = reader.readtext(img)

#             outtext_file= os.path.join(abspath, f'check/output_{i}_easyocr.txt')
#             txt_file = os.path.join(abspath, f'check/input_{i}_easyocr.txt')
#             color=(0,0,255)
#             thick=1
#             fontpath = os.path.join(abspath, "NotoSansTC-SemiBold.ttf")
#             with open(txt_file, 'w', newline='',encoding='utf-8-sig') as file:
#                 #writer = csv.writer(file)
#                 with open(outtext_file, 'w', newline='',encoding='utf-8-sig') as f:
#                     for res in result: # 畫圖
#                         f.write("#"+'{}@{}@{}'.format(*res))
#                         print(res)                                        
#                         file.write(res[1] + "\n")
#                         pos = res[0]
#                         text = res[1]
#                         for p in [(0, 1), (1, 2), (2, 3), (3, 0)]:
#                             x0,y0=pos[p[0]]
#                             x1,y1=pos[p[1]]
#                             if p == (0,1):
#                                 font = ImageFont.truetype(fontpath, 30)
#                                 img_pil = Image.fromarray(img)
#                                 draw = ImageDraw.Draw(img_pil)
#                                 r, g, b, a = 250, 0, 0, 0
#                                 draw.text((int(x0), int(y0) - 30), text, font=font, fill=(r, g, b, a))
#                                 img = np.array(img_pil)
#                             cv2.line(img, (int(x0),int(y0)), (int(x1),int(y1)), color, thick)
#             cv2.imwrite(os.path.join(abspath, f'check/bx-input_{i}.jpg'), img)
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print(f"Time：{execution_time} 秒")


if __name__ == "__main__":
    orc_parse("../../demo/demo1.jpg")