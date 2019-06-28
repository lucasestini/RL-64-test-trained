# a = [(i,j) for i in range(7) for j in range(7)]
# len(a)
# a[-1]
# a.remove((6,6))
# a[-1]
# a.remove((10,10))
# b = [(i,j) for i in range(7) for j in range(7) if (i+j)%2 == 0]
# len(b)

# def MyFunc(**kwargs):
#     x = kwargs.get('name', 'model')
#     return x

# x = MyFunc()
# x
# type(x)

# def MyFunc(**kwargs):
#     x = 'Luca'
#     x = kwargs.get('a')
#     return x
# print(MyFunc(a = 'Lucaaaaaaaaaaaaa'))
# type(x)

# class MyClass2:
#     def __init__(self, x):
#         self.att = x
#     def print(self):
#         print('MyClass2')

# obj2 = MyClass2(5)

# class MyClass:
#     def __init__(self, MyClass2):
#         self.mc2 = MyClass2

# obj = MyClass(obj2)
# obj.mc2
# obj.mc2.att
# obj.mc2.print()


# class MyClass:
#     def __init__(self, a):
#         self.first = a
#         self.__private = 10
#         self.__private2 = -23

#     def change(self):
#         self.first = 510
#         return self.__private/7

#     def play(self):
#         print(self.first)
#         self.change()
#         print(self.first)

# ist = MyClass(66)
# print(ist.play())

# import numpy as np 
# a = [(i,j) for i in range(1,1+5) for j in range(1,1+5)]
# a_np = np.array(a)
# a_np
# type(a_np)
# a_np[1][1]
# type(a_np[1][1])

# a = (1,2)
# a_np = np.array(a)
# a_np[0][0]
# a_np[0]

# a_np = np.array([a])
# a_np[0][0]
# a_np[0]

# a_np = np.array([[a]])
# a_np[0][0]
# a_np[0]

# a_np = np.array([[*a]])
# a_np
# a_np[0][0]
# a_np[0]
# a_np.flatten()
# tuple(a_np.flatten())
# a

# np.random.rand()

# c = range(1,1+100)
# np.random.choice(c)



# import random

# etrace = {}
# type(etrace)


# MOVE_LEFT = 0
# MOVE_RIGHT = 1
# MOVE_UP = 2
# MOVE_DOWN = 3
# actions = {
#     MOVE_LEFT: "move left",
#     MOVE_RIGHT: "move right",
#     MOVE_UP: "move up",
#     MOVE_DOWN: "move down"
# }

# action = random.choice(actions)
# action
counter = []
for i in range(10):
    counter.append(i)



# etrace[((3,4),action)] = 1
# etrace[((1,4),action)] = 2
# etrace[((2,2),action)] = 3
# etrace[((3,1),action)] = 4
# etrace

# for key in etrace.keys():
#     print(etrace[key])

#lista = []
#for i in range(10):
#    lista.append([i, i+1, i*i, i-1])

import numpy as np
import random
a = [0,6,-18,65,4]
a == np.min(a)
np.nonzero(a == np.max(a))
index = np.nonzero(a == np.max(a))[0]
index
type(index)
random.choice(index)


l = [[1,2,3], [4,5,6], [9,99,100]]
l[0][0].size
l = [1,2,4]
l.size
import numpy as np
l = np.array(l)
l.size
l = [[1,2], 3 , 5]
l[1]
l = [[[1,2],3], 4, 5]
l[0]
l[0][0]
l[0][0][0]



mem_size = 1000
sample_size = 32
for i, idx in enumerate(np.random.choice(range(mem_size), sample_size, replace=False)):
    print(i, idx)

l = [[1,2,3], [4,5,6], [9,99,100]]
l_np = np.array(l)
l[0][0]
l[0,0]
l_np[0,0]