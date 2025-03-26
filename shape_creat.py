

"""
形状代码生成规则
每个形状都可通过一段独特的代码来描述。 可用两个区分大小写的字母表示任意部件，一个表示其形状类型，另一个表示其颜色。不同的层级用':’ 进行分隔，空缺部件则用 '’表示。
部件代码
C:圆形
R:矩形
W:钻石
S:星星
P:针销(无色)
c:晶体
颜色代码
u:无色
r:红色
8.绿色
p.
蓝色
c:青色
m:洋红色
y:黄色
w:白色
例如
CuCuCuCu表示:无色圆形
RbRb----表示:蓝色半矩形
"""
import random
#形状集合：
shape = ['C','R','W','S','P','c','-']
#颜色集合
color = ['u','r','g','b','c','m','y','w','-']
#随机生成n层形状代码（每层四个角）

#形状结合
def shape_compose(shape_now):
    shape_code = ''
    for i in range(len(shape_now)):
        for j in range(4):
            shape_code += shape_now[i][j]
        if i == n-1:
            break
        shape_code += ':'
    return shape_code

def shape_creat(n):
    shape_now = [['' for _ in range(4)] for _ in range(n)]
    for i in range(n):#第i层代码    
        for j in range(4):
            shape_now[i][j] = random.choice(shape)+random.choice(color)
            if '-' in shape_now[i][j]:
                if i == n-1:
                    shape_now[i][j] = '--'    
                else:
                    shape_now[i][j] = 'P-'
            
            if 'P' in shape_now[i][j]:
                shape_now[i][j] = 'P-'

    shape_code = shape_compose(shape_now)
        
    
    return shape_code
#生成多少形状
#shapes_number = input('生成形状数量')
shape_number = 10
#n = int(input('请给出要生成的层数n:'))
n = 4

for i in range(shape_number):
    n = random.randint(1,4)
    print(f'{shape_creat(n)}')

