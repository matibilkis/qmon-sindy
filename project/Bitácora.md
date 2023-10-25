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