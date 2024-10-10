import matplotlib.pyplot as plt 
import numpy as np
# %matplotlib inline



from PIL import Image, ImageDraw
import random
import numpy as np
from matplotlib import backends


########### 定义颜色转换方法 ##############
import imgviz

def colored_mask(mask, save_path=None):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    # print(colormap)
    lbl_pil.putpalette(colormap.flatten())
    if save_path is not None:
        lbl_pil.save(save_path)

    return lbl_pil

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
# 设置图像和初始值
img = np.zeros((512,512), dtype=np.uint8)
line_width = 5

# 打开交互模式
plt.ion()
plt.axis('off')
plt.imshow(img, cmap='brg')

points = [] 

# def draw_line(points, width):
#   if len(points)>1:
#     x = [p[0] for p in points]
#     y = [p[1] for p in points]  
#     plt.plot(x, y, color='w', linewidth=width)


# # 新建一个函数绘制圆形线条
# def draw_circle_line(points, width):

#   x = [p[0] for p in points]
#   y = [p[1] for p in points]

#   # 如果点数大于1才绘制线
#   if len(points) > 1:
    
#     # 起始点使用小圆
#     plt.plot(x[0], y[0], 'wo', ms=width)  

#     # 绘制主线段
#     plt.plot(x[1:-1], y[1:-1], color='w', linewidth=width)
    
#     # 终点使用大圆    
#     plt.plot(x[-1], y[-1], 'ko', ms=3*width)

# # 替换 draw_line 函数
# def draw_line(points, width):
#   draw_circle_line(points, width)

# 设置线条样式为圆弧 
line_style = '-'  
line_width = 20
# 其他代码

def draw_line(points, width):

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    out = plt.plot(x, y, color='w', linewidth=width, linestyle=line_style)

    return out

while True:

    # 获取鼠标点      
    point = plt.ginput(n=1, show_clicks=True)  

    # 获取坐标并添加到点列表
    x, y = map(int, point[0])
    points.append([x,y])

    # 实时绘制线条
    out = draw_line(points, line_width)
    print(out)
    # 更新显示    
    plt.draw()

    # # 获取画布
    # fig = plt.gcf()
    #   # 赋值给a变量
    # # # 将画布转换为PIL图片
    # buffer = backends.backend_agg.FigureCanvasAgg(fig)
    # buffer.draw()
    # pil_img = buffer.tostring_rgb()
    # # print(pil_img.mode)
    # a = Image.open(io.BytesIO(pil_img))
    # print(a.mode)
    #   # 按`w`增加线粗细
    #   if plt.waitforbuttonpress():
    #     key = plt.waitforbuttonpress()
    #     if key== 'w':
    #       line_width += 1

plt.ioff()