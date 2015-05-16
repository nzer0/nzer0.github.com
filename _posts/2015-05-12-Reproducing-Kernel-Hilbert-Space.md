---
layout: post
title: Reproducing Kernel Hilbert Space
---
**설명충 주의**: _이 포스트는 설명에 너무 충실한 나머지 분량 조절에 실패했습니다._

Hilbert Space는 Inner Product가 정의된 Vector Space이다(Inner Product 정의에 의해 자연스럽게 따라서 정의되는 Norm에 대해 Complete하다는 조건이 추가로 붙는다). 기계학습에서 Hilbert Space를 처음 접했다면 Kernel Trick에 Infinite Dimension에 뭔가 어려운 이미지가 있는 녀석인데 이 정의를 보면 생각보다 간단하다(Complete 개념을 빼면). 물론 Inner Product가 고등학교에서 배우는 녀석에 비해 General하게 확장된 녀석이어서 Inner Product의 정의에 따라 굉장히 다양한 분야를 다룰 수 있고 중요하다. 

Hilbert Space중에서도 Reproducing Property를 가진 녀석들을 Reproducing Kernel Hilbert Space (RKHS) 라고 부르고, 기계학습에서 이 개념이 종종 등장한다. Reproducing Property에 대해 설명하기 전에 Hilbert Space의 한 예시로, $$L^2$$ space를 알아보자.

### $$L^p$$ space
$$L^p$$ space는 Normed Vector Space(Norm이 정의된 Vector Space)의 한 종류인데[^Banach] 여기서 $$p=2$$ 인 경우, Inner Product를 유사하게 정의할 수 있어 Hilbert Space가 되므로 잠깐 설명한다.

임의의 집합 $$\mathcal{X}$$에서 실수 집합으로의 함수 $$f: \mathcal{X} \to \mathbb{R}$$를 생각해보자. 그리고, 이 중 다음 조건을 만족하는 함수들의 집합을 생각해보자.

$$\begin{align}
\Vert f \Vert_p \equiv \left(\int_{\mathcal{X}} ~\vert f(x) \vert^p dx\right)^{\frac{1}{p}} < \infty
\end{align}$$

즉, 함수값 $$f(x)$$에 절대값을 취하고 $$p$$제곱을 해서 적분을 한 뒤 $$1/p$$승해서 얻은 값이 수렴한다는 조건이다(더 정확히는 Measure를 이용해 Lebesgue Integral로 써야 한다). 우선 아래와 같이 덧셈과 스칼라 곱셈을 정의했을 때 **이러한 함수들의 집합이 Vector Space**인 것을 확인할 수 있다(자세한 증명은 [위키](https://en.wikipedia.org/wiki/Lp_space) 참조).

$$\begin{align}
\text{Addition}&: (f+g)(x) = f(x) + g(x) \\
\text{Scalar Multiplication}&:  (\alpha f)(x) = \alpha f(x)
\end{align}$$

그리고, 위에서 정의된 $$\Vert f \Vert_p$$가 Seminorm임을 보일 수 있다. 따라서, 여기서 지지고 볶고 하면 Normed Vector Space를 얻을 수 있다.[^Seminorm] 

$$p=2$$인 경우 Norm의 정의와 유사하게 특별히 아래의 연산을 정의해볼 수 있는데,

$$\begin{align}
\langle f, g \rangle \equiv \left(\int_{\mathcal{X}} ~f(x) g(x) dx\right)^{\frac{1}{2}}
\end{align}$$

이 연산은 Inner Product가 되기위한 조건을 만족시킨다.

> **Inner Product**
> $$ \langle y,x \rangle = \langle x,y \rangle$$
> $$ \langle a x_1 +b x_2, y \rangle = a \langle x_1, y \rangle + b \langle x_2, y \rangle $$
> $$ \langle x,x \rangle \geq 0 $$

설명이 조금 길었는데, 아무튼 함수 $$f: \mathcal{X} \to \mathbb{R}$$가 우리가 보통 알던 Vector가 아닌데도 Vector Space처럼 생각할 수 있으며, 심지어 Inner Product도 계산할 수 있다는 것이 핵심이다.

### Reproducing Kernel
다시 본론으로 들어와서, Reproducing Property는 무엇인가?

> **Reproducing Property of Kernel**
> Consider Hilbert space $$H$$ of functions $$f:\mathcal{X} \to \mathbb{R}$$ and kernel function $$k:\mathcal{X} \times \mathcal{X} \to \mathbb{R}$$ such that $$k(\cdot, x) \in H, \forall x \in \mathcal{X} $$. The reproducing property is defined by,
> 
> $$\begin{align} \langle f, k(\cdot, x) \rangle_H = f(x) \end{align}, \forall x \in \mathcal{X}$$

위에서 $$L^p$$ Space를 살펴봤으니 이제 *Hilbert space of functions* 라는 말은 생소하지 않다(Hilbert Space가 꼭 $$L^2$$ Space일 필요는 없지만). Kernel 함수는 $$\mathcal{X}$$의 원소 두 개를 받아 실수 값을 결과로 내는 다변수 함수인데, 한 가지 특징이 있다면 인자 하나를 $$x$$로 고정시켰을 때 **단변수 함수** $$k(\cdot, x): \mathcal{X} \to \mathbb{R}$$가 같은 Hilbert Space 상에 있어야 한다는 것이다(그래야 Inner Product를 하지). 이때 Reproducing Property는 어떤 함수 $$f$$와 $$k(\cdot, x)$$를 Innder Product하는 게 $$f$$를 $$x$$값에 대해 Evaluation 하는 것과 같다는 것이다.


### Reproducing Kernel Hilbert Space (RKHS)
**Reproducing Kernel을 가지고 있는 Hilbert Space가 RKHS이다**. 하지만 이것은 Theorem에 의한 것이고, 제대로된 정의는 따로 있다.

> **Reproducing Kernel Hilbert Space**
> A Hilbert Space $$H$$ of functions $$f:\mathcal{X} \to \mathbb{R}$$ is called Reproducing Kernel Hilbert Space if the evaluation functional $$L_x: H \to \mathbb{R}$$ is a bounded ($$\iff$$ continuous) linear operator for all $$x \in \mathcal{X}$$

위의 정의에서 희안하게 Kernel에 대한 얘기는 한 마디도 없다. 하지만, 정의에서 출발해서 Reproducing Kernel을 이끌어낼 수 있고, 역으로 Reproducing Kernel이 존재하면 위의 정의를 이끌어낼 수 있다. 여기서는 전자의 경우를 살펴본다.

우선 정의에서 **Evaluation functional** $$L_x$$가 나오는데, 이 녀석은 함수이긴 한데 인자로 함수를 받아서 실수를 내뱉는 녀석이다. 함수가지고 Inner Product도 하는 마당에 함수를 인자로 받는 건 놀랍지도 않다. 함수가 함수를 인자로 받는다고 쓰면 자꾸 헷갈리기 때문에 이런 녀석들을 따로 functional이라고 부른다. 무튼 여기서 정의가 끝이 아니고, functinonal이면서도 $$L_x(f) = f(x)$$의 특징을 가지고 있는 녀석을 Evaluation functional이라고 부른다. 그러니까 함수를 받아서 아무 실수나 뱉는게 아니라 $$x$$ 위치에서 Evaluation한 값을 뱉는 것이다(이 때문에 그냥 $$L$$이라고 안 쓰고, $$L_x$$라고 쓴다).

우선 $$L_x$$가 Linear Operator라는 것을 확인할 수 있는데, $$f, g \in H$$의 덧셈연산은 다음과 같이 정의되고 $$(f+g)(x) = f(x) + g(x)$$ 이를 Evaluation Functional을 이용해서 쓰면 $$L_x(f+g) = L_x(f)+L_x(g)$$이다. 스칼라곱에 대해서도 같은 방법으로 하면 $$L_x$$가 Linear Operator임을 증명할 수 있다. 

$$L_x$$가 bounded operator이면 ($$L_x(f) = f(x) \lt M \Vert f \Vert_H, \exist M>0$$)

[aa](https://en.wikipedia.org/wiki/Bounded_operator#Equivalence_of_boundedness_and_continuity)
--- 

[^Banach]: 사실 $$1 \lt p \lt \infty$$면 Complete해서 Banach Space임
[^Seminorm]: Norm이 정의되기 위해서는 Zero-vector를 제외하고는 Norm값이 모두 0보다 큰 값이어야 한다. 그런데 현재 정의상으로는 $$f(x) \equiv 0$$ 외에 Lebesgue Measure 0인 함수도 적분값이 0이다. 따라서 Norm이 아닌 Seminorm이다. Seminormed Vector Space는 항상 Normed Vector Space로 만들 수 있는데, 우리의 경우 적분 값이 0이 되는 함수들을 묶어서 하나의 원소로 만든다. 수학용어로는 이렇게 묶어서 만든 녀석을 Quotient Space라고 하는데, 이는 이제 Normed Vector Space가 된다.



