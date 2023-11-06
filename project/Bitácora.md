I'm not sure i understand why we call this model oscillator... it's an oscillator in phase space, but if we understand $p=\partial_t q$ then it's not the equation of motion for the oscillator!	




![[Pasted image 20230925060516.png]]

OK.... i have the code for integrate Lorenz



Now we take a look at Suppementary Material of Pnas 2016...
###### Sparsity

![[Pasted image 20230925074351.png]]


Learning non-linear dynamics with boundary problems (PRR!)

https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.023255

https://quantum-journal.org/papers/q-2022-10-13-837/

https://arxiv.org/abs/2010.14577

TIME-DEPENDENT COEFFICIENTS
https://epubs.siam.org/doi/10.1137/18M1191944
for instance i can't learn $A(t)\cos(\omega t)$ but i can $A \cos(\omega t)$


Beyond Libland
https://arxiv.org/abs/2305.04108



El Lasso o estas cosas con el SIndy directamente no las podemos hacer, necesitamos la recurrent cell, porque esto es por un lado estocástico, y por otro lado el estado depende del measurement outcome! Entonces necesitamos una suerte de Kalman... valdría la pena inspeccionar extended Kalman?


## Implementation

![[Pasted image 20231031132052.png]]

![[Pasted image 20231031131954.png]]

* The gradients are very small and I need to enlarge the learning_rate!



![[Pasted image 20231031164443.png]]


Little roadmap:
i need to improve signal-noise ratio







### Toy-example, sindy w/ gradients in torch


it's the torch_integrate_lineal_oscillator.ipynb


I initiallize in the right parameters, then it doesn't move "much"

![[Pasted image 20231105110438.png]]

### alpha = 10.

![[Pasted image 20231105110549.png]]

![[Pasted image 20231105110601.png]]


### alpha = 0.0

![[Pasted image 20231105113014.png]]



I will free the parameters now,
beggining with 2*A_true + zeros(cubic)

![[Pasted image 20231105112315.png]]observe that there is a very small contribution (~0.1) of the frequency for the third mode...
if we needed to replicate the dynamics:

![[Pasted image 20231105112622.png]]

(ni tan mal...)


ahora a bit less "sparse" intiially---> WRONG!

![[Pasted image 20231106105350.png]]