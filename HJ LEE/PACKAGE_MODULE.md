# 알아야 할 주요 모듈, 패키지 정리

##NUMPY


##PANDA



##Matplotlib
Matplotlib은 python에서 그래프를 그릴때 사용하는 패키지
matplotlib은 2차원 그래픽 패키지이다. Matlab과 같이 커맨드 방식(matplotlib에서는 Pyplot API라고 한다)으로 그래프를 그릴 수 있으며, 커맨드 함수의 이름도 유사도록 설계되어 있다

###sample 
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,1,50)

y1 = np.cos(4*np.pi*x)
y2 = np.cos(4*np.pi*x)*np.exp(-2*x)

plt.plot(x,y1)
plt.plot(x,y2)

plt.show() #jupyter 에선 불필요


###주요 함수 요약
커맨드 방식의 함수인 Pyplot API 중 주로 사용하는 함수

plot()
subplot()
title()
xlabel()
ylabel()
axis()
xlim()
ylim()
tight_layout()
grid()
legend()
show()
figure()
text()
subplots()
