---
layout: post
title: Theano, 헷갈리는 shared variable 및 updates, givens 개념
comments: true
---

Theano를 처음 익힐 때 조금 헷갈리는 부분이 shared variable, 그리고 updates, givens 문법인데, 이런 문법이 굳이 왜 필요한지 의문이 들기 때문이다. [여기 링크](http://deeplearning.net/software/theano/tutorial/examples.html)에 설명이 잘 돼있긴 하지만 좀 더 정리해본다.


### 예제

설명을 위한 예제로 [Ordinary Least Square](https://en.wikipedia.org/wiki/Ordinary_least_squares) 문제 (Linear Regression) 

$$\begin{align}
w^* = argmin_w \sum\limits_{i=1}^n (t_i-y_i)^2, ~~~~where~~ y_i = x_i^T w
\end{align}$$ 

또는 Vector Form으로 나타내면 

$$\begin{align}
w^* = argmin_w  \Vert t-Xw \Vert^2, ~~~~where~~ X \in \mathbb{R}^{n \times d}, t \in \mathbb{R}^n
\end{align}$$ 

를 생각해보자. 사실 이 문제는 Analytic Solution이 존재하지만 여기서는 예시를 위해 Gradient Descent Learning을 하는 것으로 생각한다. 우선 Theano가 권장하는 형태로는 아래와 같이 구현할 수 있다. 

{% highlight python %}
import theano.tensor as T
import theano
# Define symbolic variables
X = T.matrix('X')
w = theano.shared([0.1, 0.1], name='w')
t = T.vector('t')

# Define Loss Expression
L = (t-X*w)**2

# Calculate Gradient Expression
dLdw = T.grad(L, w)

# Compile the training function
lr = 0.1
data_X = theano.shared([[0.1, 0.2], [0.2, 0.3], [0.1, 0.4], [0.2, 0.4]])
data_t = theano.shared([3, 3.5, 4, 4.2])
calc_output = theano.function([], L, 
		updates=[(w, w - lr*dLdw)], givens=[(X,data_X), (t,data_t)] )

for epoch in xrange(100):
	calc_output()
{% endhighlight %}

위의 코드에서 `calc_output` 함수는 실행될 때마다 Gradient Descent에 따라 $$w$$를 업데이트하고, 계산 시 `X`, `t` 변수에 각각 해당되는 데이터를 대입하여 계산한다. 따라서, `calc_output()` 함수를 실행할 때마다 학습이 진행된다.




### Givens
Givens구문은 왜 필요할까? `calc_output` 사실 아래처럼 givens 없이 구현해도 되지 않는가?

{% highlight python %}
data_X = [[0.1, 0.2], [0.2, 0.3], [0.1, 0.4], [0.2, 0.4]]
data_t = [3, 3.5, 4, 4.2]
calc_output = theano.function([X,t], L, updates=[(w, w - lr*dLdw)])

for epoch in xrange(100):
	calc_output(data_X, data_t)
{% endhighlight %}

그러니까 함수를 호출할 때 직접 인자 형태로 데이터를 넣어서 계산하는 방법말이다. 이렇게 구현했을 때 사실 CPU모드로 계산하면 별 문제가 없는데 **GPU모드를 사용할 때 단점이 드러난다. 그리고 사실 Givens 뿐만아니라, Updates와 Shared Variable도 GPU모드에서의 속도 저하 때문에 필요한 문법이다.** 

기본적으로 GPU 연산은 데이터를 주메모리에서 GPU용 메모리인 VRAM으로 옮긴 후 처리된다. 그리고 그 결과를 확인하려면 다시 VRAM에서 주메모리로 가져와야한다. 바로 이 부분이 시간이 오래 소요되는 부분이고, GPU를 이용해 프로그래밍을 할 때 첫번째로 고려해야 할게 이 메모리간 이동을 최소화 하는 것이다. 방금 수정한 코드를 GPU모드에서 실행한다면 `calc_output` 함수를 실행할 때마다 데이터를 주메모리에서 VRAM으로 옮겨서 계산하게 될 것이다. 하지만 원래의 구현을 보면 `data_X`, `data_t`는 Shared Variable로 정의되어 이미 GPU에 올라가 있는 상태이고, `calc_output` 함수를 여러번 실행하더라도 매번 옮길 필요가 없이 givens로 대입해서 쓰면 되는 것이다.

### Updates
Updates 구문은 처음 보기에는 단순히 문법적 편이성 때문에 도입된 것 같은 생각이 든다. 하지만 역시나 GPU모드에서 메모리 복사를 최소화 시키기 위한 문법이다. Updates없이 아래와 같이 구현해볼 수 있다. 

{% highlight python %}
w = T.vector('w')
calc_output = theano.function([w], [L,dLdw], givens=[(X,data_X), (t,data_t)])

weight = [0.1, 0.1]
for epoch in xrange(100):
	[L, w_grad] = calc_output(weight)
	weight = weight - lr * w_grad
{% endhighlight %}

이 방법을 GPU모드에서 쓰면 GPU에서 계산된 weight gradient `dLdw`를 매번 주메모리로 가져와야 한다. 이걸 GPU에서 수행하게 만들기 위해 Updates를 사용해야하는 것이다.


### Shared Variable
앞서 Givens, Updates를 설명하면서 Shared Variable의 필요성은 어느 정도 설명된 것 같긴 하다. 바로 GPU의 VRAM에 데이터를 올려놓기 위함이다. Theano 문서에 따르면

> **Shared Variable**: hybrid symbolic and non-symbolic variables whose value may be shared between multiple functions

이렇게 정의 돼 있어서 Hybrid라는 건 알겠는데 왜 필요한지 바로 와닿지 않는다. Symbolic Variable인데 값을 가지고 있다니, 함수형 언어에 익숙한 사람들은 필요성 뿐만 아니라 함수형 철학의 Violatoin이라고 생각할 것이다. 실제로 Updates 섹션에서 수정한 코드처럼 `w`를 theano function의 입력으로 받게 하면 Shared Variable 없이 구현할 수 있다(예제는 생략). 하지만 역시 메모리 복사 문제가 생길 뿐더러, Givens와 Updates 모두 쓸 수 없게 된다. 그러니 Shared Variable은 VRAM에 값을 유지하게 하기 위한 수단으로 이해하는 게 맞다.



### 마치며.. 
조금 자세히 Shared Variable 관련 이슈를 적어보았는데, Theano에서 가장 중요한 개념이니 잘 알고 있는게 좋다. 항상 구현할 때 Shared Variable을 우선으로 고려하고, Givens, Updates를 적극 사용하자. 하지만 Shared Variable을 어쩔 수 없이 활용하지 못하는 경우도 있다. 바로, 그래픽 카드의 VRAM 사이즈가 부족한 경우이다. 

보통 Deep Learning 모델을 학습시킬 때, Mini-batch를 쓴다고 하더라도 데이터를 전부 Shared로 올려두고, 일부분씩 사용한다. 문제는 보통 데이터가 잡아먹는 메모리가 굉장히 커서 VRAM에 다 올릴 수 있는 경우가 있다는 건데, 이때는 어쩔 수 없이 한 번씩 주메모리에서 VRAM으로 복사하는 방식으로 구현해야 한다. 예를 들어 전체 데이터 수가 100이고, Mini-batch 사이즈가 10인데, VRAM에 올릴 수 있는 한계가 50이라고 하면, Mini-batch 5번에 한 번씩 메모리 복사를 하도록 구현하면 된다. (그냥 타이탄X 같은 괴수급 VRAM의 그래픽 카드를 달자..)









