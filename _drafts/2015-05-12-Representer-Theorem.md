---
layout: post
title: Representer Theorem
---

### Representer Theorem:

$$n$$ 개의 Training 데이터 $$(x_i, y_i)_{i=1}^n$$ 가 주어졌고, 모델의 candidate 집합 (Hypothesis Set)을 어떠한 Reproducing Kernel Hilbert Space $$H$$ 로 두었을 때 최적의 모델 $$f^*$$를 구하기 위해,  아래와 같이 에러 $$E$$와 Regularizer $$g$$의 합으로 나타나는 Loss Minimization 식을 쓸 수 있다.

$$
\begin{align}
f^* = argmin_{f \in H_k} ( E((x_i,y_i,f(x_i))_{i=1}^n) + g(\Vert f \Vert) )
\end{align}
$$

여기서 항상 최적해를 $$f^*=\sum\limits_{i=1}^n \alpha_i \phi(x_i)$$ 형태로 나타낼 수 있다는게 Representer Theorem이다. 

증명은 Reproducing Kernel Hilbert Space의 속성을 이용하면 간단한데, 