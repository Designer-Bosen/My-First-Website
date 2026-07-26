# Stats 100B - Introduction to Mathematical Statistics

**Instructor:** Professor Nicolas Christou

---

## Distribution Summary

$$\textbf{Discrete Distributions}$$

| Distribution | Probability Mass Function | Mean | Variance | Moment Generating Function |
|--------------|---------------------------|:----:|:--------:|----------------------------|
| Binomial | $P(X=x)=\binom{n}{x}p^x(1-p)^{n-x}$<br>$x=0,1,\ldots,n$ | $np$ | $np(1-p)$ | $\left(pe^t+(1-p)\right)^n$ |
| Geometric | $P(X=x)=(1-p)^{x-1}p$<br>$x=1,2,\ldots$ | $\frac{1}{p}$ | $\frac{1-p}{p^2}$ | $\frac{pe^t}{1-(1-p)e^t}$ |
| Negative Binomial | $P(X=x)=\binom{x-1}{r-1}p^r(1-p)^{x-r}$<br>$x=r,r+1,\ldots$ | $\frac{r}{p}$ | $\frac{r(1-p)}{p^2}$ | $\left(\frac{pe^t}{1-(1-p)e^t}\right)^r$ |
| Hypergeometric | $P(X=x)=\dfrac{\binom{r}{x}\binom{N-r}{n-x}}{\binom{N}{n}}$<br>$x=0,\ldots,n$ if $n\le r$<br>$x=0,\ldots,r$ if $n>r$ | $\dfrac{nr}{N}$ | $n\dfrac{r}{N}\dfrac{N-r}{N}\dfrac{N-n}{N-1}$ | Fairly complicated |
| Poisson | $P(X=x)=\dfrac{\lambda^xe^{-\lambda}}{x!}$<br>$x=0,1,\ldots$ | $\lambda$ | $\lambda$ | $\exp\!\left(\lambda(e^t-1)\right)$ |

$$\textbf{Continuous Distributions}$$

| Distribution | Probability Density Function | Mean | Variance | Moment Generating Function |
|--------------|------------------------------|:----:|:--------:|----------------------------|
| Uniform | $f(x)=\dfrac1{b-a}$<br>$a\le x\le b$ | $\dfrac{a+b}{2}$ | $\dfrac{(b-a)^2}{12}$ | $\dfrac{e^{tb}-e^{ta}}{t(b-a)}$ |
| Gamma | $f(x)=\dfrac{x^{\alpha-1}e^{-x/\beta}}{\beta^\alpha\Gamma(\alpha)}$<br>$\alpha,\beta>0,\ x\ge0$ | $\alpha\beta$ | $\alpha\beta^2$ | $(1-\beta t)^{-\alpha}$ |
| Exponential | $f(x)=\lambda e^{-\lambda x}$<br>$\lambda>0,\ x\ge0$ | $\dfrac1\lambda$ | $\dfrac1{\lambda^2}$ | $\left(1-\dfrac{t}{\lambda}\right)^{-1}$ |
| Beta | $f(x)=\dfrac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$<br>$\alpha,\beta>0,\ 0\le x\le1$ | $\dfrac{\alpha}{\alpha+\beta}$ | $\dfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |  |
| Normal | $f(x)=\dfrac1{\sigma\sqrt{2\pi}}e^{-\frac12(\frac{x-\mu}{\sigma})^2}$<br>$-\infty < x < \infty$ | $\mu$ | $\sigma^2$ | $e^{\mu t+\frac{\sigma^2t^2}{2}}$ |

---

# Lec 1: Random vectors and properties

## Mean and variance of a random vector

Let $\mathbf{Y}=\begin{pmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{pmatrix}$ be a random vector with $E\mathbf{Y}=\begin{pmatrix} EY_1 \\ EY_2 \\ \vdots \\ EY_n \end{pmatrix}=\begin{pmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_n \end{pmatrix}$. The covariance matrix of $\textbf{Y}$ denoted with $\Sigma = cov[Y]$ is defined as follows:

$$
\begin{aligned}
cov[\mathbf{Y}]&=E(\mathbf{Y} - \boldsymbol{\mu})(\mathbf{Y} - \boldsymbol{\mu})^T \\
&= E\begin{pmatrix} \mathbf{Y}_1 - \boldsymbol{\mu}_1 \\ \mathbf{Y}_2 - \mathbf{\mu}_2 \\ \vdots \\ \mathbf{Y}_n - \boldsymbol{\mu}_n \end{pmatrix} \begin{pmatrix} \mathbf{Y}_1 - \boldsymbol{\mu}_1 & \mathbf{Y}_2 - \boldsymbol{\mu}_2 & \dots & \mathbf{Y}_n - \boldsymbol{\mu}_n \end{pmatrix} \\
&= \begin{pmatrix}
(\mathbf{Y}_1-\boldsymbol{\mu}_1)^2 &
(\mathbf{Y}_1-\boldsymbol{\mu}_1)(\mathbf{Y}_2-\boldsymbol{\mu}_2) &
\cdots &
(\mathbf{Y}_1-\boldsymbol{\mu}_1)(\mathbf{Y}_n-\boldsymbol{\mu}_n) \\
(\mathbf{Y}_2-\boldsymbol{\mu}_2)(\mathbf{Y}_1-\boldsymbol{\mu}_1) &
(\mathbf{Y}_2-\boldsymbol{\mu}_2)^2 &
\cdots &
(\mathbf{Y}_2-\boldsymbol{\mu}_2)(\mathbf{Y}_n-\boldsymbol{\mu}_n) \\
\vdots & \vdots & \ddots & \vdots \\
(\mathbf{Y}_n-\boldsymbol{\mu}_n)(\mathbf{Y}_1-\boldsymbol{\mu}_1) &
(\mathbf{Y}_n-\boldsymbol{\mu}_n)(\mathbf{Y}_2-\boldsymbol{\mu}_2) &
\cdots &
(\mathbf{Y}_n-\boldsymbol{\mu}_n)^2
\end{pmatrix} \\
&= \begin{pmatrix}
\sigma_{11} & \sigma_{12} & \cdots & \sigma_{1n} \\
\sigma_{21} & \sigma_{22} & \cdots & \sigma_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{n1} & \sigma_{n2} & \cdots & \sigma_{nn}
\end{pmatrix} = \boldsymbol{\Sigma}
\end{aligned}
$$

So $\boldsymbol{\Sigma}$ is the covariance matrix of $\mathbf{Y}$. It is symmetric and positive definite. Suppose $Y_1, \dots, Y_n$ are independent identically distributed (i.i.d.) random variables. This means that $E[Y_i] = \mu, i = 1, \dots, n$; $var[Y_i] = \sigma^2, i = 1, \dots, n$ and $cov[Y_i, Y_j] = 0, i \neq j$. Then $E[\mathbf{Y}]=\mu \begin{pmatrix} 1 \\ 1 \\ \vdots \\ 1\end{pmatrix} $, $var[\mathbf{Y}]=\sigma^2I_n$


## Result 1 
Expected value and variance of a linear combination of $\mathbf{Y}$. Let $\mathbf{a}^T=\begin{pmatrix} a_1 & a_2 & \dots & a_n \end{pmatrix}$ be a vector of constants and let $q=\mathbf{a}^T\mathbf{Y}$. Then $E[q]=E[\mathbf{a}^T\mathbf{Y}]=\mathbf{a}^TE[\mathbf{Y}]=\mathbf{a}^T\boldsymbol{\mu}$. The variance of q can be found as follows:

$$
\begin{aligned}
var[q]=E[(q-\boldsymbol{\mu}_q)^2]
&=E[(\mathbf{a}^T\mathbf{Y} - \mathbf{a}^T\boldsymbol{\mu})(\mathbf{a}^T\mathbf{Y} - \mathbf{a}^T\boldsymbol{\mu})] \\
&= \mathbf{a}^T E[(\mathbf{Y} - \boldsymbol{\mu})(\mathbf{Y} - \boldsymbol{\mu})^T] \mathbf{a} \\
&= \mathbf{a}^T \boldsymbol{\Sigma} \mathbf{a}
\end{aligned}
$$

Note: $q$ is a scalar and therefore its variance should be a scalar and not a matrix. ($\mathbf{a}^T \boldsymbol{\Sigma} \mathbf{a}$ is $1 \times 1$) We can also express the variance of a linear combination as follows:

$$
\begin{aligned}
var[\sum_{i=1}^na_iY_i]&=cov[\sum_{i=1}^na_iY_i, \sum_{j=1}^na_jY_j] \\
&= \sum_{i=1}^n \sum_{j=1}^n a_ia_j cov[Y_i, Y_j] \\
&= \sum_{i=1}^n a_i^2 var[Y_i] + \sum_{i=1}^n \sum_{j \neq i}^n a_i a_j cov[Y_i, Y_j] \\
&= \sum_{i=1}^n a_i^2 var[Y_i] + 2\sum_{i=1}^{n-1} \sum_{j > i}^n a_i a_j cov[Y_i, Y_j]
\end{aligned}
$$


## Result 2
Let $\mathbf{A}$ be a $p \times n$ matrix of constants. $\mathbf{Q} = \mathbf{A}\mathbf{Y}$ is a $p \times 1$ vector and therefore its variance should be a $p \times p$ matrix. $E[\mathbf{Q}]=E[\mathbf{A}\mathbf{Y}]=\mathbf{A}E[Y]=\mathbf{A}\boldsymbol{\mu}$. For the variance of $\mathbf{Q}$ use the definition of the covariance matrix of a random vector.

$$
\begin{aligned}
var[\mathbf{Q}]&=E[(\mathbf{Q} - \boldsymbol{\mu}_\mathbf{Q})(\mathbf{Q} - \boldsymbol{\mu}_\mathbf{Q})^T] \\
&= E[(\mathbf{A}\mathbf{Y} - \mathbf{A}\boldsymbol{\mu})(\mathbf{A}\mathbf{Y} - \mathbf{A}\boldsymbol{\mu})^T] \\
&= \mathbf{A} E[(\mathbf{Y} - \boldsymbol{\mu})(\mathbf{Y} - \boldsymbol{\mu})^T] \mathbf{A}^T \\
&= \mathbf{A} \boldsymbol{\Sigma} \mathbf{A}^T
\end{aligned}
$$


## Result 3 - Expectation of a quadratic expression
Let $\mathbf{Y}$ be a random vector $n \times 1$ and let $\mathbf{A}$ be an $n \times n$ matrix of constants. Consider the quadratic expression $\mathbf{Y}^T\mathbf{A}\mathbf{Y}$. To find the expected value of this quadratic expression, we will use properties of hte trace of a square matrix. We can do this because $\mathbf{Y}^T\mathbf{A}\mathbf{Y}$ isi a scalar. We will also need this result: $E[\mathbf{Y}\mathbf{T}^T]=\boldsymbol{\Sigma} + \boldsymbol{\mu}\boldsymbol{\mu}^T$. 

$$
\begin{aligned}
E[\mathbf{Y}^T\mathbf{A}\mathbf{Y}]=E[tr(\mathbf{Y}^T\mathbf{A}\mathbf{Y})]&=E[tr(\mathbf{A}\mathbf{Y}\mathbf{Y}^T)] \\
&= tr(E[\mathbf{A}\mathbf{Y}\mathbf{Y}^T]) \\
&= tr(\mathbf{A}E[\mathbf{Y}\mathbf{Y}^T]) \\
&= tr(\mathbf{A}(\boldsymbol{\Sigma} + \boldsymbol{\mu}\boldsymbol{\mu}^T)) \\
&= tr(\mathbf{A}\boldsymbol{\Sigma} + \mathbf{A}\boldsymbol{\mu}\boldsymbol{\mu}^T) \\
&= tr(\mathbf{A}\boldsymbol{\Sigma}) + tr(\mathbf{A}\boldsymbol{\mu}\boldsymbol{\mu}^T) \\
&= tr(\mathbf{A}\boldsymbol{\Sigma}) + \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu} \\
\end{aligned}
$$


## Other Results
$cov[\mathbf{a}^T\mathbf{Y}, \mathbf{b}^T\mathbf{Y}] = \mathbf{a}^T\boldsymbol{\Sigma}\mathbf{b}$. This is a scalar

$cov[\mathbf{A}^T\mathbf{Y}, \mathbf{B}^T\mathbf{Y}] = \mathbf{A}^T\boldsymbol{\Sigma}\mathbf{B}$. This is a matrix


---


# Lec 2: A note on expectation, independence, and exponential families

1. Let $X$ be a continuous random variable. Then $E[X]=\int_x xf(x)dx$
2. Suppose we want to find the expectation of a function of $X$. Let $Y = g(X)$. Show that $E[g(X)]=\int_xg(x)f(x)dx$
    <div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
    **Proof**:

    One way to compute $E[Y]$ is to find the pdf of $Y$. Use the method of cdf. Begin with the cdf of Y.
    $$F_Y(y) = P[Y \leq y] = P[g(X) \leq y] = P[X \leq w(y)] = F_X(w(y))$$

    Take derivative on both sides w.r.t. $y$ to get

    $$f_Y(y) = w'(y) f_x(w(y)) \qquad E[Y]=\int_y y w'(y) f_x(w(y)) dy$$
    
    Back to the proof. Let $I = \int_xg(x)f(x)dx$, $y = g(x)$ and solve for $x$. We get $x = w(y)$, $\frac{dx}{dy} = w'(y)$. Transform the integral $I$ in terms of y: 

    $$I = \int_xg(x)f(x)dx = \int_y y f(w(y)) w'(y) dy = E[Y]$$
    </div>

## Kernel Function 

Let $X \sim \Gamma(\alpha, \beta), \alpha > 0, \beta > 0, x > 0$. Then $f(x) = \frac{x^{\alpha-1} e^{-\frac{x}{\beta}}}{\Gamma(\alpha)\beta^\alpha}$. The part of the pdf $x^{\alpha-1} e^{-\frac{x}{\beta}}$ is called kernel function.

**Definition (gamma function)**: $\Gamma(\alpha) = \int_0^\infty x^{\alpha-1} e^{-x} dx$

**Properties**: 
- $\Gamma(\alpha) = (\alpha - 1)\Gamma(\alpha - 1)$
- $\Gamma(\alpha) = (\alpha - 1)! \quad$ (if $\alpha$ is an integer)

### Example
Let $X \sim exp(1)$, then $f(x)=e^{-x}$. Find $E[X^3]=\int_0^\infty x^3 e^{-x}dx$
To evaluate this integral, we can use the kernel function of gamma distribution. 
$$\int_0^\infty \frac{x^{4-1} e^{-\frac{x}{1}}}{\Gamma(4)1^4} dx = 1 \Rightarrow \int_0^\infty x^3 e^{-x}dx = \int_0^\infty x^{4-1} e^{-\frac{x}{1}} = \Gamma(4) = 3! = 6$$

## Note on Independence 
Let $X$, $Y$ be random variables with joint pdf $f(x,y)$. Then $X$, $Y$ are independent if $f(x,y)=f(x)f(y)$ 

Note: To find the marginal pdf, use $f(x) = \int_y f(x,y)dy$ and $f(y) = \int_x f(x,y)dx$

### Theorem
Let $X$, $Y$ be independent random variables. Then $E[XY]=E[X]E[Y]$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">

**Proof**:

$XY$ is a function of $X$ and $Y$. Therefore, using the expectation of a function of $x$ and $y$ $E[g(x,y)] = \int_x \int_y g(x,y) f(x,y) dx dy$, we get:

$$
\begin{aligned}
E[XY] &= \int_x \int_y xy f(x,y) dx dy \\
&= \int_x \int_y xy f(x)f(y) dx dy \\
&= (\int_x x f(x) dx)( \int_y y f(y) dy) \\
&= E[X]E[Y]
\end{aligned}
$$

and $cov[X,Y]=E[XY] - E[X]E[Y] = 0$
</div>

### Corollary: 
Let $X$, $Y$ be independent random variables, and let $g(x)$ and $h(y)$ be functions of $x$ and $y$ alone respectively. Then $E[g(X)h(Y)] = E[g(X)]E[h(Y)]$

### Theorem
Let $g(x)$ be a function of $X$ alone and $h(y)$ be a function of $Y$ alone. Then $X$, $Y$ are independent iff $f(x,y)=g(x)h(y)$.

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">

**Proof**:

Let $c = \int_{-\infty}^\infty g(x) dx$ and $d = \int_{-\infty}^\infty h(y) dy$. 

$$
cd = (\int_{-\infty}^\infty g(x) dx)(\int_{-\infty}^\infty h(y) dy) 
= \int_{-\infty}^\infty \int_{-\infty}^\infty g(x)h(y) dx dy 
= \int_{-\infty}^\infty \int_{-\infty}^\infty f(x,y) dx dy = 1
$$

Then marginal distribution of $X$ and $Y$ are given as 

$$f(x) = \int_{-\infty}^\infty f(x,y) dy = \int_{-\infty}^\infty g(x)h(y) dy = \int_{-\infty}^\infty h(y) dy \cdot g(x) = d \cdot g(x)$$

$$f(y) = \int_{-\infty}^\infty f(x,y) dx = \int_{-\infty}^\infty g(x)h(y) dx = \int_{-\infty}^\infty g(x) dx \cdot h(y) = c \cdot h(y)$$

Finally, 

$$f(x,y) = c \cdot d \cdot g(x)h(y) = f(x)f(y) \Rightarrow \text{X and Y are independent}$$

</div>


## Exponential families

A probability density function or probability mass function is called an exponential family if it can be expressed as 

$$f(x \mid \boldsymbol{\theta}) = h(x)c(\boldsymbol{\theta}) \exp\Big(\sum_{i=1}^k w_i(\boldsymbol{\theta})t_i(x)\Big)$$

Note: let $d$ be $dim(\boldsymbol{\theta})$. If $d=k$, we have full exponential family; 
if $d < k$, we have curved exponential family.

### Example
Let $X \sim bin(n,p)$ with $n$ fixed. 

$$
\begin{aligned}
p(x) &= \binom{n}{x} p^x(1-p)^{n-x} \\
&= \binom{n}{x} \Big(\frac{p}{1-p}\Big)^x(1-p)^n \\
&= \binom{n}{x} (1-p)^n \exp{\Big(\log{\Big(\frac{p}{1-p}\Big)^x}\Big)} \\
&= \binom{n}{x} (1-p)^n \exp{\Big(x\log{\Big(\frac{p}{1-p}\Big)}\Big)} 
\end{aligned}
$$

Therefore, this pmf is an exponential family with

$h(x) = \binom{n}{x}, c(p) = (1-p)^n, t_1(x) = x, w_1(p) = \log{\Big(\frac{p}{1-p}\Big)}$

### Theorem
Suppose a random variable $X$ has a pdf or pmf that can be expressed in the form of exponential family. Then, 
- (a) $E\Big[\sum_{i=1}^k \frac{\partial w_i(\boldsymbol{\theta})}{\partial \theta_j}t_i(x)\Big] = -\frac{\partial}{\partial \theta_j}\log c(\boldsymbol{\theta})$
- (b) $var\Big[\sum_{i=1}^k \frac{\partial w_i(\boldsymbol{\theta})}{\partial \theta_j}t_i(x)\Big] = -\frac{\partial^2}{\partial \theta_j^2}\log c(\boldsymbol{\theta}) - E\Big[\sum_{i=1}^k \frac{\partial^2 w_i(\boldsymbol{\theta})}{\partial \theta_j^2}t_i(x)\Big]$


<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">

**Proof of (a)**:

$$
\begin{aligned}
\int_x f(x \mid \boldsymbol{\theta})dx &= 1 \\
\int_x h(x)c(\boldsymbol{\theta}) \exp\Big(\sum_{i=1}^k w_i(\boldsymbol{\theta})t_i(x)\Big) dx &= 1
\end{aligned}
$$

Differentiate both sides w.r.t. $\theta_j$:

$$
\begin{aligned}
&\int_x h(x) \frac{\partial c(\boldsymbol{\theta})}{\partial \theta_j} \exp\Big(\sum_{i=1}^k w_i(\boldsymbol{\theta})t_i(x)\Big) dx \\
+ &\int_x h(x)c(\boldsymbol{\theta}) \sum_{i=1}^k \frac{\partial w_i(\boldsymbol{\theta})}{\partial \theta_j}t_i(x)\exp\Big(\sum_{i=1}^k w_i(\boldsymbol{\theta})t_i(x)\Big) dx = 0
\end{aligned}
$$

Multiply the first integral by $\frac{c(\boldsymbol{\theta})}{c(\boldsymbol{\theta})}$ and note that $\frac{\partial \log c(\boldsymbol{\theta})}{\partial \theta_j} = \frac{\partial c(\boldsymbol{\theta})}{\partial \theta_j} \frac{1}{c(\boldsymbol{\theta})}$

$$
\begin{aligned}
&\int_x h(x) \frac{\partial c(\boldsymbol{\theta})}{\partial \theta_j} \exp\Big(\sum_{i=1}^k w_i(\boldsymbol{\theta})t_i(x)\Big) \frac{c(\boldsymbol{\theta})}{c(\boldsymbol{\theta})} dx \\
+ &\int_x h(x)c(\boldsymbol{\theta}) \sum_{i=1}^k \frac{\partial w_i(\boldsymbol{\theta})}{\partial \theta_j}t_i(x)\exp\Big(\sum_{i=1}^k w_i(\boldsymbol{\theta})t_i(x)\Big) dx = 0
\end{aligned}
$$

After rearranging we get

$$
\begin{aligned}
\int_x \sum_{i=1}^k \frac{\partial w_i(\boldsymbol{\theta})}{\partial \theta_j} t_i(x) h(x)c(\boldsymbol{\theta}) \exp\Big(\sum_{i=1}^k w_i(\boldsymbol{\theta})t_i(x)\Big) &dx = \\
- \frac{\partial \log c(\boldsymbol{\theta})}{\partial \theta_j} \int_x h(x)c(\boldsymbol{\theta}) \exp\Big(\sum_{i=1}^k w_i(\boldsymbol{\theta})t_i(x)\Big) &dx
\end{aligned}
$$

Or

$$E\Big[ \sum_{i=1}^k \frac{\partial w_i(\boldsymbol{\theta})}{\partial \theta_j} t_i(x) \Big] = - \frac{\partial}{\partial \theta_j} \log c(\boldsymbol{\theta})$$

To prove statement (b), differentiate a second time and rearrange

</div>

### Example 
Let $X \sim \text{Poisson}(\lambda)$. Use the theorem above to show that $E[X] = \lambda$ and $var[X] = \lambda$ 

$$p(x)=\frac{\lambda^xe^{-\lambda}}{x!} = \frac{1}{x!}e^{-\lambda}e^{\log\lambda^x} = \frac{1}{x!}e^{-\lambda}e^{x\log\lambda}$$

So this pmf is an exponential family with 

$h(x) = \frac{1}{x!}, c(\lambda) = e^{-\lambda}, t_1(x) = x, w_1(\lambda) = \log\lambda$

$$E[\frac{1}{\lambda}X] = -(-1) \quad \Rightarrow \quad E[X] = \lambda$$

$$
var[\frac{1}{\lambda}X] = 0 - E[-\frac{1}{\lambda^2}X] \\
\quad \Rightarrow \quad \frac{1}{\lambda^2} var[X] = \frac{1}{\lambda^2} E[X] \\
\quad \Rightarrow \quad var[X] = E[X] = \lambda
$$


---


# Lec 3: Moment generating functions

### Definition (MGF)
$$
M_X(t) = E[e^{tX}]=
\begin{cases}
\displaystyle \sum_x e^{tx}p(x), & \text{if } X \text{ is discrete},\\[1.2em]
\displaystyle \int_x e^{tx}f(x) dx, & \text{if } X \text{ is continuous}.
\end{cases}
$$

Aside:

$$
e^x=1 + \frac{x}{1!} + \frac{x^2}{2!} + \frac{x^3}{3!} + \dots \qquad 
e^x=1 + \frac{tx}{1!} + \frac{(tx)^2}{2!} + \frac{(tx)^3}{3!} + \dots
$$

Let $X$ be a discrete random variable.

$$
\begin{aligned}
M_t = \sum_x e^{tX}p(x) &= \sum_x \Big[ 1 + \frac{tx}{1!} + \frac{(tx)^2}{2!} + \frac{(tx)^3}{3!} + \dots \Big] p(x) \\
&= \sum_x p(x) + \sum_x \frac{tx}{1!} p(x) + \sum_x \frac{(tx)^2}{2!} p(x) + \sum_x \frac{(tx)^3}{3!} p(x) + \dots
\end{aligned}
$$

To find the $k_{th}$ moment, simply evaluate the $k_{th}$ derivative of $M_X(t)$ at $t=0$.

$$E[X]^k = [M_X(t)]_{t=0}^{k_{th} \text{derivative}}$$

First moment: $M_X'(t) = \sum_x x p(x) + \sum_x \frac{2t}{2!}x^2 p(x) + \dots \qquad M_X'(0) = \sum_x x p(x) = E[X]$

Second moment: $M_X'(t) = \sum_x x^2 p(x) + \sum_x \frac{6t}{3!}x^3 p(x) + \dots \qquad M_X''(0) = \sum_x x^2 p(x) = E[X^2]$

Or from direct differentiation of MGF from the definition and evaluate at $t=0$.

$$
\begin{aligned}
M_X(t) &= E[E^{tX}] \quad \Rightarrow \quad M_X(0) = E[1] = 1 \\
M_X'(t) &= \frac{\partial M_X(t)}{\partial t} = E[Xe^{tX}] \quad \Rightarrow \quad M_X'(0) = E[X] \\
M_X''(t) &= \frac{\partial^2 M_X(t)}{\partial t^2} = E[X^2e^{tX}] \quad \Rightarrow \quad M_X''(0) = E[X^2]
\end{aligned}
$$

## Corollary 
Instead of differentiating $M_X(t)$, we can differentiate $\log[M_X(t)]$ and evaluate the first and second derivative at $t=0$. This will give $E[X]$ and $var[X]$.

$$
\begin{aligned}
\Psi(t) &= \log[M_X(t)] \\

\Psi'(t) &= \frac{M_X'(t)}{M_X(t)} \quad \Rightarrow \quad \frac{M_X'(0)}{M_X(0)} = E[X] \\

\Psi''(t) &= \frac{M_X''(t)M_X(t) - (M_X'(t))^2}{(M_X(t))^2} \quad \Rightarrow \quad \frac{E[X^2]\cdot 1 - E[X]^2}{1^2} = var[X]
\end{aligned}
$$

### MGF of binomial random variable

Let $X \sim bin(n,p)$

$$
\begin{aligned}
M_X(t) &= E[e^{tX}] \\
&= \sum_{x=0}^n e^{tx} \binom{n}{x} p^x(1-p)^{n-x} \\
&= \sum_{x=0}^n \binom{n}{x} (pe^t)^x(1-p)^{n-x} \qquad \text{applying binomial theorem} \\
&= (pe^t + 1 - p)^n
\end{aligned}
$$

### MGF of Poisson random variable

Let $X \sim \text{Poisson}(\lambda)$

$$
\begin{aligned}
M_X(t) &= E[e^{tX}] \\
&= \sum_{x=0}^\infty e^{tx} \frac{\lambda^x e^{-\lambda}}{x!} \\
&= e^{-\lambda} \sum_{x=0}^\infty \frac{(\lambda e^{t})^x}{x!} \\
&= e^{-\lambda} e^{\lambda e^t} \\
&= e^{\lambda(e^t - 1)}
\end{aligned}
$$

### MGF of gamma random variable

Let $X \sim \Gamma(\alpha, \beta)$

$$
\begin{aligned}
M_X(t) &= E[e^{tX}] \\
&= \int_{x=0}^\infty e^{tx} \frac{x^{\alpha-1} e^{-\frac{x}{\beta}}}{\Gamma(\alpha)\beta^\alpha} dx \\
&= \int_{x=0}^\infty \frac{x^{\alpha-1} e^{-x (\frac{1}{\beta} - t)}}{\Gamma(\alpha)\beta^\alpha} dx \\
&= \frac{\beta^{*\alpha}}{\beta^\alpha} \int_{x=0}^\infty \frac{x^{\alpha-1} e^{-\frac{x}{\beta^*}}}{\Gamma(\alpha)\beta^{*\alpha}} dx \qquad \text{where } \beta^* = \frac{\beta}{1 - \beta t} \\
&= \frac{\beta^{*\alpha}}{\beta^\alpha} \\
&= (1-\beta t)^{-\alpha} \\
\end{aligned}
$$

### MGF of exponential random variable

Let $X \sim exp(\lambda)$. The exponential distribution is a special case of $\Gamma(\alpha, \beta)$ with $\alpha = 1$ and $\beta = \frac{1}{\lambda}$, therefore, $M_X(t) = (1-\frac{t}{\lambda})^{-1}$

### MGF of standard normal random variable

Let $Z \sim N(0,1)$

$$
\begin{aligned}
M_Z(t) &= E[e^{tZ}] \\
&= \int_{-\infty}^\infty e^{tz} \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}} dz \\
&= e^{\frac{t^2}{2}} \int_{-\infty}^\infty \frac{1}{\sqrt{2\pi}} e^{-\frac{(z-t)^2}{2}} dz \\
&= e^{\frac{t^2}{2}}
\end{aligned}
$$

### Theorem 
Let $X$, $Y$ be independent random variables with moment generating functions $M_X(t)$, $M_Y(t)$ respectively. Then, the moment generating function of the sum of these two random variables is equal to the product of the individual moment generating functions:

$$M_{X+Y}(t) = M_X(t)M_Y(t)$$

**Proof**: $M_{X+Y}(t) = E[e^{t(X+Y)}] = E[e^{tX} \cdot e^{tY}] \overset{\text{indep}}{=} E[e^{tX}]E[e^{tY}] = M_X(t)M_Y(t)$ 

### Example 

$$X \sim bin(n_1, p), Y \sim bin(n_2, p), X \perp\!\!\!\perp Y$$

$$M_{X+Y}(t) = M_X(t)M_Y(t) = (pe^t + 1 - p)^{n_1 + n_2}$$

$$X + Y \sim bin(n_1 + n_2, p)$$

### Example 

$$X \sim \text{Poisson}(\lambda_1), Y \sim \text{Poisson}(\lambda_2), X \perp\!\!\!\perp Y$$

$$M_{X+Y}(t) = M_X(t)M_Y(t) = e^{(\lambda_1 + \lambda_2)(e^t - 1)}$$

$$X + Y \sim \text{Poisson}(\lambda_1 + \lambda_2)$$

## Properties of moment generating functions
Let $X$ be a random variable with moment genearting function $M_X(t) = E[e^{tX}]$, and $a$, $b$ are constants
- $M_{X+a}(t) = E[e^{t(X+a)}] = e^{ta} E[e^{tX}] = e^{at}M_X(t)$
- $M_{bX}(t) = E[e^{tbX}] = M_X(bt)$
- $M_{\frac{X+a}{b}}(t) = e^{\frac{a}{b}t} E[e^{\frac{t}{b}X}] = e^{\frac{a}{b}t} M_X(\frac{t}{b})$

### Example - MGF of normal random variable

Use the moment generating function of $Z \sim N(0,1)$ to find the moment generating function of $X \sim N(\mu, \sigma^2)$

$$Z = \frac{X - \mu}{\sigma} \quad \Rightarrow \quad X = \mu + \sigma Z$$

$$
\begin{aligned}
M_X(t) = M_{\mu + \sigma Z}(t) &= E[e^{t(\mu + \sigma Z)}] \\
&= e^{t\mu} E[e^t\sigma Z] \\
&= e^{t\mu} M_Z(\sigma t) \\
&= e^{t\mu} e^{\frac{1}{2}t^2\sigma^2} \\
&= e^{t\mu + \frac{1}{2}t^2\sigma^2}
\end{aligned}
$$

### Example 
Suppose $X$, $Y$ are independent random variables. Find the distribution of $X+Y$, where $X \sim N(\mu_1, \sigma_1), Y \sim N(\mu_2, \sigma_2)$

$$M_{X+Y}(t) = M_X(t)M_Y(t) = e^{t\mu_1 + \frac{1}{2}t^2\sigma_1^2} e^{t\mu_2 + \frac{1}{2}t^2\sigma_2^2} = e^{t(\mu_1 + \mu_2) + \frac{1}{2}t^2(\sigma_1^2 + \sigma_2^2)}$$

$$E[X+Y] = \mu_1 + \mu_2 \qquad var[X+Y] = \sigma_1^2 + \sigma_2^2$$

$$X+Y \sim N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$$

### Example
Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$. Use moment generating functions to find the distribution.

#### (a). $T = X_1 + X_2 + \dots + X_n$

$$
\begin{aligned}
M_T(t) = M_{X_1 + X_2 + \dots + X_n}(t) 
&= M_{X_1}(t)M_{X_2}(t) \dots M_{X_n}(t) \\
&= \Big( M_{X_i}(t) \Big)^n \\
&= \Big( e^{t\mu + \frac{1}{2}t^2\sigma^2} \Big)^n \\
&= e^{tn\mu + \frac{1}{2}t^2n\sigma^2} \\
\end{aligned}
$$

$$T \sim N(n\mu, n\sigma^2)$$


#### (b). $\bar{X} = \frac{\sum_{i=1}^n X_i}{n}$

$$M_\bar{X}(t) = M_{\frac{T}{n}}(t) = M_T(\frac{t}{n}) = e^{t\mu + \frac{1}{2}t^2\frac{\sigma^2}{n}}$$

$$\bar{X} \sim N(\mu, \frac{\sigma^2}{n})$$


---


# Lec 4: Functions of random variables 

## Method of cdf (Single variable transformation)

### Example 
Let $X \sim \Gamma(\alpha, \beta)$. Find the distribution of $Y = cX, c > 0$, with method of cdf.

$$
\begin{aligned}
F_Y(y) &= P(Y \leq y) \\
&= P(cX \leq y) \\
&= P(X \leq \frac{y}{c}) \\
&= F_X(\frac{y}{c}) \\
f_Y(y) &= \frac{1}{c} f_X(\frac{y}{c}) \\
&= \frac{1}{c} \frac{\frac{y}{c}^{\alpha-1} e^{-\frac{y}{c\beta}}}{\Gamma(\alpha)\beta^\alpha} \\
&= \frac{y^{\alpha-1} e^{-\frac{y}{c\beta}}}{\Gamma(\alpha)(c\beta)^\alpha}
\end{aligned}
$$

Therefore, $Y \sim \Gamma(\alpha, c\beta)$

### Example
Let $Z \sim N(0,1)$. Find the pdf of $X = Z^2$.

$$
\begin{aligned}
F_X(x) &= P(X \leq x) \\
&= P(Z^2 \leq x) \\
&= P(-\sqrt{x} \leq Z \leq \sqrt{x}) \\
&= P(Z \leq \sqrt{x}) - P(Z \leq -\sqrt{x}) \\
&= F_Z{\sqrt{x}} - F_Z{-\sqrt{x}} \\
f_X(x) &= \frac{1}{2}x^{-\frac{1}{2}}f_Z(\sqrt{x}) + \frac{1}{2}x^{-\frac{1}{2}}f_Z(-\sqrt{x}) \\
&= \frac{1}{2}x^{-\frac{1}{2}} \frac{1}{\sqrt{2\pi}}e^{-\frac{x}{2}} + \frac{1}{2}x^{-\frac{1}{2}}\frac{1}{\sqrt{2\pi}}e^{-\frac{x}{2}} \\
&= \frac{x^{-\frac{1}{2}}e^{-\frac{x}{2}}}{\sqrt{2\pi}} \\
&= \frac{x^{-\frac{1}{2}}e^{-\frac{x}{2}}}{\Gamma{\frac{1}{2}}2^{\frac{1}{2}}}
\end{aligned}
$$

Therefore, $Z^2 \sim \Gamma(\frac{1}{2}, 2) = \chi^2_1$

## Method of transformation (Single variable transformation)

Let $X$ be a continuous random variable with pdf $f(x)$. Let $Y=g(X)$ be a monotone function (either increasing or decreasing), then there is an one-to-one transformation. The pdf of Y(X) is given by 

$$f_Y(y) = f_X[g^{-1}(y)]\Big| \frac{d}{dy}g^{-1}(y) \Big|$$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">

**Proof**:

**Case 1: decreasing function**

$$
\begin{aligned}
F_Y(y) &= P(Y \leq y) \\
&= P(g(x) \leq y) \\
&= P(X \leq g^{-1}(y)) \\
&= F_X(g^{-1}(y)) \\
f_Y(y) &= \frac{\partial}{\partial y}F_Y(y) \\
&= \frac{\partial}{\partial y}F_X(g^{-1}(y)) \\
&= f_X(g^{-1}(y)) \frac{d}{dy}g^{-1}(y) \quad (\text{positive} \cdot \text{positive})
\end{aligned}
$$

**Case 2: increasing function**

$$
\begin{aligned}
F_Y(y) &= P(Y \leq y) \\
&= P(g(x) \leq y) \\
&= P(X > g^{-1}(y)) \\
&= 1 - F_X(g^{-1}(y)) \\
f_Y(y) &= \frac{\partial}{\partial y}F_Y(y) \\
&= \frac{\partial}{\partial y}(1 - F_X(g^{-1}(y))) \\
&= - f_X(g^{-1}(y)) \frac{d}{dy}g^{-1}(y) \quad (-\text{positive} \cdot \text{negative})
\end{aligned}
$$

Therefore $f_Y(y) = f_X[g^{-1}(y)]\Big| \frac{d}{dy}g^{-1}(y) \Big|$

**Results**

Let $X$ be a continuous random variable and let $Y=g(X)$
- If $g(X)$ is increasing, then $F_Y(y) = F_X[g^{-1}(y)]$
- If $g(X)$ is decreasing, then $F_Y(y) = 1 - F_X[g^{-1}(y)]$

</div>

### Example 
Let $X \sim \Gamma(\alpha, \beta)$. Find the distribution of $Y = cX, c > 0$, with method of transformation.

$$
\begin{aligned}
g(x) &= cX \quad \Rightarrow \quad g^{-1}(y)=\frac{y}{c} \\
f_Y(y) &= f_X[g^{-1}(y)]\Big| \frac{d}{dy}g^{-1}(y) \Big| \\
&= f_x(\frac{y}{c})\cdot \frac{1}{c} \\
&= \frac{y^{\alpha-1} e^{-\frac{y}{c\beta}}}{\Gamma(\alpha)(c\beta)^\alpha}
\end{aligned}
$$

### Example 
Let $X \sim unif(0,1)$, $Y = ln(X)$

$$f(x)=1, \quad F(x)=x$$
$$y = g(x) = \log(y) \quad \Rightarrow \quad g^{-1}(y)=e^{-y}$$
$$f_Y(y) = f_X(e^{-y}) \Big| \frac{d}{dy}e^{-y}\Big| = |-e^{-y}| = e^{-y}$$

Therefore, $Y \sim exp(1)$

### Example
Let $X \sim exp(\lambda)$, $Y = 1 - e^{-\lambda X}$

$$x = g^{-1}(y) = \frac{\log(1-y)}{-\lambda}$$
$$f_Y(y) = f_X(g^{-1}(y)) \Big| \frac{d}{dy} g^{-1}(y) \Big| = \lambda e^{-\lambda \frac{\log(1-y)}{-\lambda}} \Big| \frac{1}{(1-y)\lambda} \Big| = 1$$

Therefore, $Y \sim unif(0,1)$

### Example 
Let $X \sim \Gamma(n, \beta)$ (n is an integer). Find the pdf of $Y = \frac{1}{X}$ (inverted gamma distribution)

$$x = g^{-1}(y) = \frac{1}{y}$$
$$f_Y(y) = f_X(g^{-1}(y)) \Big| \frac{d}{dy} g^{-1}(y) \Big| = \frac{(\frac{1}{y})^{n-1}e^{-\frac{1}{y\beta}}}{(n-1)!\beta^n} \cdot \Big| -\frac{1}{y^2} \Big| = \frac{(y)^{-(n+1)}e^{-\frac{1}{y\beta}}}{(n-1)!\beta^n}$$

### Theorem 
Let $X$ be a continuous random variable with pdf $f(x)$ and suppose $Y = g(X)$ is monotone in the intervals $A_1, A_2, \dots, A_k$. Then 

$$f_Y(y) = \sum_{i=1}^k f_X[g_i^{-1}(y)] \Big| \frac{d}{dy} g_i^{-1}(y)\Big|$$

### Example
Let $X \sim N(0,1)$, $Y = X^2$. Find the pdf of $Y$.

$$A_1 = (-\infty, 0), g_1^{-1}(y)=-\sqrt{y}$$
$$A_2 = (0, \infty), g_2^{-1}(y)=\sqrt{y}$$

$$
\begin{aligned}
f_Y(y) &= f_X[g_1^{-1}(y)]\Big| \frac{d}{dy}g_1^{-1}(y) \Big| + f_Y(y) = f_X[g_2^{-1}(y)]\Big| \frac{d}{dy}g_2^{-1}(y) \Big| \\
&= \frac{1}{\sqrt{2\pi}}e^{-\frac{y}{2}} |-\frac{1}{2}y^{-\frac{1}{2}}| + \frac{1}{\sqrt{2\pi}}e^{-\frac{y}{2}} |\frac{1}{2}y^{-\frac{1}{2}}| \\
&= \frac{y^{-\frac{1}{2}}e^{-\frac{y}{2}}}{\sqrt{2\pi}} \\
&= \frac{y^{\frac{1}{2} - 1}e^{-\frac{y}{2}}}{\Gamma(\frac{1}{2})2^\frac{1}{2}}
\end{aligned}
$$

Therefore, $Y \sim \Gamma(\frac{1}{2}, 2) = \chi^2_1$


## Joint probability distribution of functions of random variables

Let $X$, $Y$ be continuous random variables with joint pdf $f_{X,Y}(x,y)$. Suppose $U=g_1(X,Y)$ and $V=g_2(X,Y)$. Assume that the transformation is one-to-one. To find the joint pdf of $U$, $V$, we follow this procedure:
- We need the joint pdf of $X$ and $Y$.
- Solvle the equations $U=g_1(X,Y)$ and $V=g_2(X,Y)$ for $x$ and $y$ in terms of $u$ and $v$ to get $x=h_1(u,v)$ and $y=h_2(u,v)$.
- Compute the Jacobian: 
$\mathbf{J_1} = \begin{vmatrix} \frac{\partial g_1}{\partial x} & \frac{\partial g_1}{\partial y} \\ \frac{\partial g_2}{\partial x} & \frac{\partial g_2}{\partial y} \end{vmatrix}$ and 
$\mathbf{J_2} = \begin{vmatrix} \frac{\partial h_1}{\partial u} & \frac{\partial h_1}{\partial v} \\ \frac{\partial h_2}{\partial u} & \frac{\partial h_2}{\partial v} \end{vmatrix}$

Finally, to find the joint pdf of $U$, $V$, use the following result:

$$
\begin{aligned}
f_{U,V}(u,v) &= f_{X,Y}(x=h_1,y=h_2)|\mathbf{J_1}|^{-1} \quad &(\text{inverse of absolute value of the Jacobian})\\
OR \quad f_{U,V}(u,v)  &= f_{X,Y}(x=h_1,y=h_2)|\mathbf{J_2}| \quad &(\text{absolute value of the Jacobian})
\end{aligned}
$$

### Example 
Let $X_1$ and $X_2$ be independent exponential random variables with parameters $\lambda_1$ and $\lambda_2$ respectively. Find the joint pdf of $U = X_1 + X_2$ and $V = X_1 - X_2$.

Since $X_1$ and $X_2$ are independent, the joint pdf of $X_1$ and $X_2$ is

$$f_{X_1,X_2}(x_1,x_2)=f_{X_1}(x_1)f_{X_2}(x_2)=\lambda_1e^{-\lambda_1x_1}\cdot\lambda_2e^{-\lambda_2x_2}=\lambda_1\lambda_2e^{-\lambda_1x_1-\lambda_2x_2}$$

Solve for $X_1$ and $X_2$ in terms of $U$ and $V$:

$$
\begin{cases}
U = X_1 + X_2 \\
V = X_1 - X_2
\end{cases} \quad \Rightarrow \quad
\begin{cases}
X_1 = \frac{U+V}{2}\\
X_2 = \frac{U-V}{2}
\end{cases}
$$

Compute the Jacobian $\mathbf{J_2}$

$$\mathbf{J_2} = \begin{vmatrix} \frac{\partial x_1}{\partial u} & \frac{\partial x_1}{\partial v} \\ \frac{\partial x_2}{\partial u} & \frac{\partial x_2}{\partial v} \end{vmatrix}
=\begin{vmatrix} \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2} & -\frac{1}{2} \end{vmatrix}$$

Finally, find the joint pdf of $U$ and $V$

$$f_{U,V}(u,v) = \lambda_1\lambda_2e^{\lambda_1\frac{U+V}{2}+\lambda_2\frac{U-V}{2}} \cdot |-\frac{1}{2}|$$

### Example 
Suppose $X$ and $Y$ are independent random variables with $X \sim \Gamma(\alpha_1, \beta)$ and $Y \sim \Gamma(\alpha_2, \beta)$. Compute the joint pdf of $U=X+Y$ and $V=\frac{X}{X+Y}$ and find the distribution of $U$ and the distribution of $V$. Also show that $U$, $V$ are independent.

Because $X$, $Y$ are independent, the joint pdf of $X$ and $Y$ is the product of the two marginal pdfs:

$$f_{X,Y}(x,y) = f_X(x)f_Y(y) 
= \frac{x^{\alpha_1-1} e^{-\frac{x}{\beta}}}{\Gamma(\alpha_1)\beta^{\alpha_1}} 
\frac{y^{\alpha_2-1} e^{-\frac{y}{\beta}}}{\Gamma(\alpha_2)\beta^{\alpha_2}}
= \frac{x^{\alpha_1-1} y^{\alpha_2-1} e^{-\frac{x+y}{\beta}}}{\Gamma(\alpha_1)\Gamma(\alpha_2)\beta^{\alpha_1+\alpha_2}}
$$

Then solve the equations to get $x=uv$ and $y=u(1-v)$

Compute Jacobian $\mathbf{J_1} = \begin{vmatrix} \frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\ \frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} \end{vmatrix} = \begin{vmatrix} 1 & 1 \\ \frac{y}{(x+y)^2} & -\frac{x}{(x+y)^2}  \end{vmatrix} = -\frac{1}{x+y} = -\frac{1}{u}$

Finally to find the joint pdf of $U$, $V$ use $x=uv$ and $y=u(1-v)$ in the joint pdf of $X$,$Y$: 

$f_{U,V}(u,v) = \frac{(uv)^{\alpha_1-1}[u(1-v)]^{\alpha_2-1}e^{-\frac{u}{\beta}}u}{\Gamma(\alpha_1)\Gamma(\alpha_2)\beta^{\alpha_1+\alpha_2}}$, multiply by $\frac{\Gamma(\alpha_1+\alpha_2)}{\Gamma(\alpha_1+\alpha_2)}$ and rearrange to get:

$$f_{U,V}(u,v) = \frac{u^{\alpha_1+\alpha_2-1}e^{-\frac{u}{\beta}}}{\Gamma(\alpha_1+\alpha_2)\beta^{\alpha_1+\alpha_2}} \times \frac{v^{\alpha_1-1}(1-v)^{\alpha_2-1}\Gamma(\alpha_1+\alpha_2)}{\Gamma(\alpha_1)\Gamma(\alpha_2)} = \frac{u^{\alpha_1+\alpha_2-1}e^{-\frac{u}{\beta}}}{\Gamma(\alpha_1+\alpha_2)\beta^{\alpha_1+\alpha_2}} \times \frac{v^{\alpha_1-1}(1-v)^{\alpha_2-1}}{B(\alpha_1,\alpha_2)}$$

where $B(\alpha_1,\alpha_2) = \int_0^1 v^{\alpha_1-1}(1-v)^{\alpha_2-1} dv = \frac{\Gamma(\alpha_1)\Gamma(\alpha_2)}{\Gamma(\alpha_1+\alpha_2)}$ is the Beta function.

Therefore, $U \perp\!\!\!\perp V$, $U \sim \Gamma(\alpha_1+\alpha_2, \beta)$, $V \sim \text{Beta}(\alpha_1, \alpha_2)$

### Example - Distribution of the ratio of normal random variables
Let $X$ and $Y$ be independent standard normal random variables. Find the joint pdf of $U+X+Y$ and $V=X-Y$.

$$f_{X,Y}(x,y) = f_X(x)f_Y(y) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}} \cdot \frac{1}{\sqrt{2\pi}}e^{-\frac{y^2}{2}}$$

$\mathbf{J_1} = \begin{vmatrix} \frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\ \frac{\partial v}{\partial x} & \frac{\partial v}{\partial y} \end{vmatrix} = \begin{vmatrix} 1 & 1 \\ 1 & -1  \end{vmatrix} = -2$

$$
\begin{aligned}
f_{U,V}(u,v) &= \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{u+v}{2})^2} \cdot \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{u+v}{2})^2} \cdot |-\frac{1}{2}| \\
&= \frac{1}{\sqrt{2}\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{u}{\sqrt{2}})^2} \cdot \frac{1}{\sqrt{2}\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{v}{\sqrt{2}})^2}
\end{aligned}
$$

Therefore, $U \sim N(0,\sqrt{2})$, $V \sim N(0,\sqrt{2})$

### Note 1
Suppose we are interested only in finding the pdf of $U = g_1(X,Y)$. We can still use the previous method by defining a function $V = g_2(X,Y)$. We then obtain the joint pdf of $U$ and $V$ and from there we find hte marginal distribution of $U$.

### Note 2
If the transformation is not one-to-one, we use teh following theorem:

$$f_{U,V}(u,v) = \sum_{i=1}^k f_{X,Y}(h_{1i}(u,v),h_{2i}(u,v)) |J_i|$$

### Example - Distribution of the ratio of normal random variables
Let $X$ and $Y$ be independent standard normal random variables. Let $U=\frac{X}{Y}$ and $V=|Y|$. Find the joint pdf of $U$, $V$ and then find the marginal distribution of $U$.

$U=\frac{X}{Y}$ is one-to-one but $V=|Y|$ is not.

Let $A_1 = {(x,y) \mid y < 0}$ and $A_2 = {(x,y) \mid y > 0}$, then 

$$
\begin{aligned}
&h_{11}(u,v) = -uv \quad &h_{12}(u,v) = uv \\
&h_{21}(u,v) = -v \quad &h_{22}(u,v) = v 
\end{aligned}
$$

$$\mathbf{J_2} = \begin{vmatrix} \frac{\partial h_{11}}{\partial u} & \frac{\partial h_{12}}{\partial v} \\ \frac{\partial h_{21}}{\partial u} & \frac{\partial h_{22}}{\partial v} \end{vmatrix} = \begin{vmatrix} -v & -u \\ 0 & 1 \end{vmatrix}= v$$

$$
\begin{aligned}
f_{X,Y}(x,y) &= f_X(x)f_Y(y) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}} \cdot \frac{1}{\sqrt{2\pi}}e^{-\frac{y^2}{2}} = \frac{1}{2\pi}e^{-\frac{x^2 + y^2}{2}} \\
f_{U,V}(u,v) &= \frac{1}{2\pi}e^{-\frac{(-uv)^2 + (-v)^2}{2}} \cdot v + \frac{1}{2\pi}e^{-\frac{(uv)^2 + (v)^2}{2}} \cdot v \\
&= \frac{v}{\pi}e^{-\frac{(u^2+1)v^2}{2}} \\
f_U(u) &= \int_0^\infty \frac{v}{\pi}e^{-\frac{(u^2+1)v^2}{2}} dv \\
&= \int_0^\infty \frac{2}{\pi}e^{-\frac{(u^2+1)v^2}{2}} dv^2 \\
\quad \text{let } t &= \frac{(u^2+1)v^2}{2}, dt = (u^2+1)v dv \quad \Rightarrow \quad v dv = \frac{dt}{u^2+1})= \\
f_U(u) &= \int_0^\infty \frac{1}{\pi}e^{-t}\frac{dt}{u^2+1} \\
&= \frac{1}{\pi(u^2+1)} \int_0^\infty e^{-t} dt \\
&= \frac{1}{\pi(u^2+1)} \quad (\text{Cauchy Distribution})
\end{aligned}
$$


---


# Lec 5: Joint moment generating functions

Let $\mathbf{X} = \begin{pmatrix} X_1 \\ X_2 \\ \vdots \\ X_n \end{pmatrix}$ be a random vector and let  $\mathbf{t} = \begin{pmatrix} t_1 \\ t_2 \\ \vdots \\ t_n \end{pmatrix}$ be a  vector of real values. The joint moment generating function of $X$ is defined as $M_\mathbf{X}(\mathbf{t}) = E[e^{\mathbf{t}^T \mathbf{X}}] = E[\exp(\sum_{i=1}^n t_ix_i)]$.

## Multinomial distribution
A sequence of $n$ independent experiments is performed and each experiment can result in one of $r$ possible outcomes with probability $p_1, p_2, \dots, p_n$ with $\sum_{i=1}^r p_i = 1$. Let X_i be the number of $n$ experiments that result in outcome $i$ for $i = 1,2,\dots,r$ with $x_1+x_2+\dots+x_r=n$. Then, $P(X_1=x_1,X_2=x_2,\dots,X_r=x_r) = \frac{n!}{x_1!x_2!\dots x_r!}p_1^{x_1}p_2^{x_2}\dots p_r^{x_r}$. Denote the multinomial distribution with $\mathbf{X} \sim M(n,\mathbf{p})$.

Joint moment generating function of $\mathbf{X} \sim M(n,\mathbf{p})$.

$$
\begin{aligned}
M_\mathbf{X}(\mathbf{t}) = E[e^{\mathbf{t}^T \mathbf{X}}] &= \sum_{x_1}\sum_{x_2}\dots \sum_{x_n} e^{\mathbf{t}^T \mathbf{X}}\frac{n!}{x_1!x_2!\dots x_r!}p_1^{x_1}p_2^{x_2}\dots p_r^{x_r} \\
&= \sum_{x_1}\sum_{x_2}\dots \sum_{x_n} \frac{n!}{x_1!x_2!\dots x_r!}(p_1e^{t_1})^{x_1}(p_2e^{t_2})^{x_2}\dots (p_re^{t_r})^{x_r} \\
&= (p_1e^{t_1} + p_2e^{t_2} + \dots + p_re^{t_r})^n \quad (\text{multinomial theorem})
\end{aligned}
$$

### Example 
Suppose a fair die is rolled $n=15$ times. the probability that we observed #1 twice, #2 three times, #3 once, #4 twice, #5 three times, #6 four times is 

$$P(X_1=2,X_2=3,X_3=1,X_4=2,X_5=3,X_6=4) = \frac{15!}{2!3!1!2!3!4!}(\frac{1}{6})^2(\frac{1}{6})^3(\frac{1}{6})^1(\frac{1}{6})^2(\frac{1}{6})^3(\frac{1}{6})^4$$

### Theorem
Let $M_i(\mathbf{t}) = \frac{\partial M_\mathbf{X}(\mathbf{t})}{\partial t_i}$, 
$M_{ii}(\mathbf{t}) = \frac{\partial^2 M_\mathbf{X}(\mathbf{t})}{\partial t_i^2}$, and
$M_{ij}(\mathbf{t}) = \frac{\partial^2 M_\mathbf{X}(\mathbf{t})}{\partial t_i \partial t_j}$.

Then, $E[X_i] = M_i(\mathbf{0})$, $E[X_i^2] = M_{ii}(\mathbf{0})$, $E[X_iX_j] = M_{ij}(\mathbf{0})$

### Example 
Let $n=2$ then let $\mathbf{X}=\begin{pmatrix} X_1 \\ X_2 \end{pmatrix}$ and $\mathbf{t}=\begin{pmatrix} t_1 \\ t_2 \end{pmatrix}$. Then, $M_\mathbf{X}(\mathbf{t}) = E[e^{\mathbf{t}^T \mathbf{X}}] = \int\int e^{\mathbf{t}^T \mathbf{X}} f(x_1,x_2) dx_1 dx_2$

- $M_1(\mathbf{t}) = \int\int x_1 e^{t_1x_1 + t_2x_2} f(x_1,x_2) dx_1 dx_2$
- $M_1(\mathbf{0}) = \int\int x_1 f(x_1,x_2) dx_1 dx_2 = \int x_1 f(x_1) dx_1 = E[X_1]$
- $M_2(\mathbf{t}) = \int\int x_2 e^{t_1x_1 + t_2x_2} f(x_1,x_2) dx_1 dx_2$
- $M_2(\mathbf{0}) = \int\int x_2 f(x_1,x_2) dx_1 dx_2 = \int x_2 f(x_2) dx_2 = E[X_2]$
- $M_{11}(\mathbf{t}) = \int\int x_1^2 e^{t_1x_1 + t_2x_2} f(x_1,x_2) dx_1 dx_2$
- $M_{11}(\mathbf{0}) = \int\int x_1^2 f(x_1,x_2) dx_1 dx_2 = \int x_1^2 f(x_1) dx_1 = E[X_1^2]$
- $M_{22}(\mathbf{t}) = \int\int x_2^2 e^{t_1x_1 + t_2x_2} f(x_1,x_2) dx_1 dx_2$
- $M_{22}(\mathbf{0}) = \int\int x_2^2 f(x_1,x_2) dx_1 dx_2 = \int x_2^2 f(x_2) dx_2 = E[X_2^2]$
- $M_{12}(\mathbf{t}) = \int\int x_1x_2 e^{t_1x_1 + t_2x_2} f(x_1,x_2) dx_1 dx_2$
- $M_{12}(\mathbf{0}) = \int\int x_1x_2 f(x_1,x_2) dx_1 dx_2 = E[X_1X_2]$
- $var[X_1] = E[X_1^2] - E[X_1]^2 = M_{11}(\mathbf{0}) - M_1(\mathbf{0})^2$
- $var[X_2] = E[X_2^2] - E[X_2]^2 = M_{22}(\mathbf{0}) - M_2(\mathbf{0})^2$
- $cov[X_1, X_2] = E[X_1X_2] - E[X_1]E[X_2] = M_{12}(\mathbf{0}) - M_1(\mathbf{0}) M_2(\mathbf{0})$

### Corollary
Let $\psi(\mathbf{t}) = \log M_X(\mathbf{t})$, 
$\psi_i(\mathbf{t}) = \frac{\partial \psi_\mathbf{X}(\mathbf{t})}{\partial t_i}$,
$\psi_{ii}(\mathbf{t}) = \frac{\partial^2 \psi_\mathbf{X}(\mathbf{t})}{\partial t_i^2}$,
$\psi_{ij}(\mathbf{t}) = \frac{\partial^2 \psi_\mathbf{X}(\mathbf{t})}{\partial t_it_j}$

Then $E[X_i] = \psi_i(\mathbf{0})$, $var[X_i] = \psi_{ii}(\mathbf{0})$, $cov[X_i, X_j] = \psi_{ij}(\mathbf{0})$.

### Example
Consider the multinomial probability distribution $\mathbf{X} \sim M(n,\mathbf{p})$ with joint moment generating function $M_\mathbf{X}(\mathbf{t}) = (p_1e^{t_1} + p_2e^{t_2} + \dots + p_re^{t_r})^n$

$$
\begin{aligned}
\psi(\mathbf{t}) &= n \log(p_1e^{t_1} + p_2e^{t_2} + \dots + p_re^{t_r}) \\
\psi_1(\mathbf{t}) &= \frac{np_1e^{t_1}}{p_1e^{t_1} + p_2e^{t_2} + \dots + p_re^{t_r}} \\
\psi_1(\mathbf{0}) &= np_1 = E[X_1] \quad (\text{binomial}) \\
\psi_{11}(\mathbf{t}) &= \frac{ np_1e^{t_1} (p_1e^{t_1} + p_2e^{t_2} + \dots + p_re^{t_r}) - np_1^2e^{2t_1}}{(p_1e^{t_1} + p_2e^{t_2} + \dots + p_re^{t_r})^2} \\
\psi_{11}(\mathbf{0}) &= np_1(1 - p_1) = var[X_1] \quad (\text{binomial})
\end{aligned}
$$

### Theorem
Let $\mathbf{X} = \begin{pmatrix} \mathbf{Y} \\ \mathbf{Z} \end{pmatrix}$. The marginal moment generating function of $\mathbf{Y}(\mathbf{Z})$ is the moment generating function of $\mathbf{X}$ ignoring the vector $\mathbf{Z}(\mathbf{Y})$. This is expressed as $M_\mathbf{Y}(\mathbf{u}) = M_\mathbf{X}(\mathbf{u}, \mathbf{0})$ and $M_\mathbf{Z}(\mathbf{v}) = M_\mathbf{X}(\mathbf{0}, \mathbf{v})$, where $\mathbf{t} = \begin{pmatrix} \mathbf{u} \\ \mathbf{v} \end{pmatrix}$.

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

$$M_\mathbf{X}(\mathbf{t}) = E[e^{\mathbf{t}^T \mathbf{X}}] = E[e^{\mathbf{u}^T \mathbf{Y} + \mathbf{v}^T \mathbf{Z}}] = E[e^{\sum u_iY_i + \sum v_iZ_i}]$$

Set $v_iZ_i = 0$

$$E[e^{\sum u_iY_i}] = E[e^{\mathbf{u}^T \mathbf{Y}}] = M_\mathbf{Y}(\mathbf{u}) = M_\mathbf{X}(\mathbf{u}, \mathbf{0})$$

Set $u_iY_i = 0$

$$E[e^{\sum v_iZ_i}] = E[e^{\mathbf{v}^T \mathbf{Z}}] = M_\mathbf{Z}(\mathbf{v}) = M_\mathbf{X}(\mathbf{0}, \mathbf{v})$$

</div>

### Example 
Consider the multinomial probability distributino $\mathbf{X} \sim M(n, \mathbf{p})$ with joint moment generating function 

$$M_\mathbf{X}(\mathbf{t}) = (p_1e^{t_1} + p_2e^{t_2} + \dots + p_re^{t_r})^n$$

Find the marginal moment generating function of $X_1$.

$$M_\mathbf{X}(t_1) = M_\mathbf{X}(t_1, 0, \dots, 0) = (p_1e^{t_1} + p_2 + \dots + p_r)^n = (p_1e^{t_1} + 1 - p_1)^n$$

Therefore $X_1 \sim bin(n,p_1)$

### Theorem 
If $\mathbf{Y}$ and $\mathbf{Z}$ are independent, then 
$M_\mathbf{X}(\mathbf{t}) = M_\mathbf{Y}(\mathbf{u})M_\mathbf{Z}(\mathbf{v})$

**Proof**: $M_\mathbf{X}(\mathbf{t}) = E[e^{\mathbf{t}^T \mathbf{X}}] = E[e^{\mathbf{u}^T \mathbf{Y} + \mathbf{v}^T \mathbf{Z}}] = E[e^{\mathbf{u}^T \mathbf{Y}}] E[e^{\mathbf{v}^T \mathbf{Z}}] = M_\mathbf{Y}(\mathbf{u})M_\mathbf{Z}(\mathbf{v})$

### Example
Let $\mathbf{X} = (X_1, X_2, X_3)$ has joint moment generating function 

$$M_\mathbf{X}(t_1,t_2,t_3) = (1-t_1+2t_2)^{-4}(1-t_1+3t_3)^{-3}(1-t_1)^{-2}$$

- Find the moment generating function of $(X_1, X_3)$
    $$M_{X_1,X_3}(t_1,t_3) = M_\mathbf{X}(t_1,0,t_3) = (1-t_1)^{-6}(1-t_1+3t_3)^{-3}$$
- Find the moment generating function of $X_1$ and $X_3$, are they independent?
    $$M_{X_1}(t_1) = M_\mathbf{X}(t_1,0,0) = (1-t_1)^{-9}$$
    $$M_{X_3}(t_3) = M_\mathbf{X}(t_1,0,0) = (1+3t_3)^{-3}$$
    Since $M_{X_1,X_3}(t_1,t_3) \neq M_{X_1}(t_1)M_{X_3}(t_3)$, $X_1$ and $X_3$ are not independent

### Example
Let $X$ and $Y$ be independent normal random variables, each with mean $\mu$ and variance $\sigma^2$ 

- Consider the random quantities $X+Y$ and $X-Y$. Find the moment generating function of $X+Y$ and the moment generating function of $X-Y$.
    $$
    \begin{aligned}
    M_{X+Y}(t) &= M_X(t)M_Y(t) = e^{t\mu + \frac{1}{2}t^2\sigma^2}e^{t\mu + \frac{1}{2}t^2\sigma^2} = e^{t2\mu + \frac{1}{2}t^22\sigma^2} \qquad &X+Y \sim N(2\mu, 2\sigma^2) \\
    M_{X-Y}(s) &= M_X(s)M_Y(-s) = e^{s\mu + \frac{1}{2}s^2\sigma^2}e^{-s\mu + \frac{1}{2}(-s)^2\sigma^2} = e^{\frac{1}{2}s^22\sigma^2} \qquad &X+Y \sim N(0, 2\sigma^2)
    \end{aligned}
    $$

- Find the joint moment generating function of $(X+Y, X-Y)$.
    $$
    \begin{aligned}
    M{X+Y,X-Y}(t,s) &= E[e^{t(X+Y)+s(X-Y)}] \\
    &= E[e^{(t+s)X+(t-s)Y}] \\
    &= M_X(t+s)M_Y(t-s) \\
    &= e^{(t+s)\mu + \frac{1}{2}(t+s)^2\sigma^2} e^{(t-s)\mu + \frac{1}{2}(t-s)^2\sigma^2} \\
    &= e^{t2\mu + \frac{1}{2}t^22\sigma^2} e^{\frac{1}{2}s^22\sigma^2} \\
    &= M_{X+Y}(t)M_{X-Y}(s)
    \end{aligned}
    $$

    Therefore, $X+Y$ and $X-Y$ are independent.


---


# Lec 6: Multivariate normal distribution

A random vector $\mathbf{Y} = (Y_1, Y_2, \dots, Y_n)^T$ with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ follows multivariate normal distribution denoted $\mathbf{Y} \sim N_n(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ has pdf given by

$$f(\mathbf{Y}) = \frac{1}{(2\pi)^\frac{n}{2}} |\boldsymbol{\Sigma}|^{-\frac{1}{2}}e^{-\frac{1}{2}(\mathbf{Y} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{Y} - \boldsymbol{\mu})}$$

If $\mathbf{Y} = (Y_1, Y_2)$ follows a bivariate normal distribution, then its pdf can be expressed as

$$f(y_1,y_2) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}} \exp\Big[ -\frac{1}{2(1-\rho^2)} \Big[ 
\Big(\frac{y_1-\mu_1}{\sigma_1}\Big)^2 + \Big(\frac{y_2-\mu_2}{\sigma_2}\Big)^2 
- 2\rho\Big(\frac{y_1-\mu_1}{\sigma_1}\Big) \Big(\frac{y_2-\mu_2}{\sigma_2}\Big) \Big] \Big]$$

where $\boldsymbol{\Sigma} = \begin{pmatrix} \sigma_1^2 & \sigma{12} \\ \sigma_{21} & \sigma_2^2 \end{pmatrix}$, $\rho = cor[Y_1, Y_2] = \frac{cov[Y_1, Y_2]}{\sigma_1\sigma_2}$, and $\sigma_{12}=\sigma_{21}=\rho\sigma_1\sigma_2$.

## Moment generating function - multivariate normal distribution

### Result 1

Suppose $\mathbf{Z} \sim N_n(\mathbf{0}, \mathbf{I})$. Since $Z_1, Z_2, \dots, Z_n$ are independent. The joint moment generating function of $\mathbf{Z}$ is $M_\mathbf{Z}(\mathbf{t}) = e^{\frac{1}{2}\mathbf{t}^T\mathbf{t}}$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

$$
f(z) = \prod_{i=1}^n f(z_i) 
= \prod_{i=1}^n \frac{1}{\sqrt{2\pi}} e^{-\frac{z_i^2}{2}} 
= \frac{1}{(2\pi)^{\frac{n}{2}}} e^{-\frac{\sum_{i=1}^n z_i^2}{2}}
= \frac{1}{(2\pi)^{\frac{n}{2}}} |\mathbf{I}|^{-\frac{1}{2}} e^{-\frac{1}{2} (z-0)^T \mathbf{I}^{-1} (z-0)}
$$

$$
\begin{aligned}
M_\mathbf{Z}(\mathbf{t}) 
= E[e^{\mathbf{t}^T \mathbf{Z}}]
= E[e^{t_1Z_1 + \dots + t_nZ_n}]
= E[e^{t_1Z_1}] \dots E[e^{t_nZ_n}] 
&= M_{Z_1}(t_1) \dots M_{Z_n}(t_n) \\
&= e^{\frac{1}{2}t_1^2} \dots e^{\frac{1}{2}t_n^2} \\
&= e^{\frac{1}{2}\sum_{i=1}^n t_i^2} \\
&= e^{\frac{1}{2}\mathbf{t}^T\mathbf{t}}
\end{aligned}
$$

</div>

### Result 2

Suppose $\mathbf{Y} \sim N_n(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then $\mathbf{Z} = \boldsymbol{\Sigma}^{-\frac{1}{2}} (\mathbf{Y} - \boldsymbol{\mu}) \sim N_n(\mathbf{0}, \mathbf{I})$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

Since $\boldsymbol{\Sigma}$ is symmetric, there exists a spectral decomposition such that $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda} \mathbf{Q}^T$, where $\boldsymbol{\Lambda} = diag(\lambda_i)$  $\mathbf{Q} = (e_1, \dots, e_n)$ are orthonormal vectors, and $\mathbf{Q}\mathbf{Q}^T = \mathbf{Q}^T\mathbf{Q} = \mathbf{I}$

**Properties**: (Assume $\boldsymbol{\Sigma}$ is positive definite)
- $\boldsymbol{\Sigma}^\frac{1}{2} = \mathbf{Q}\boldsymbol{\Lambda}^\frac{1}{2} \mathbf{Q}^T$ is a square matrix and $\boldsymbol{\Sigma}^\frac{1}{2}\boldsymbol{\Sigma}^\frac{1}{2}\boldsymbol{\Sigma}$, $(\boldsymbol{\Sigma}^\frac{1}{2})^T = \boldsymbol{\Sigma}^\frac{1}{2}$
- $\boldsymbol{\Sigma}^{-\frac{1}{2}} = \mathbf{Q}\boldsymbol{\Lambda}^{-\frac{1}{2}} \mathbf{Q}^T$ is a square matrix and $\boldsymbol{\Sigma}^{-\frac{1}{2}}\boldsymbol{\Sigma}^{-\frac{1}{2}}\boldsymbol{\Sigma}$, $(\boldsymbol{\Sigma}^{-\frac{1}{2}})^T = \boldsymbol{\Sigma}^{-\frac{1}{2}}$

Continue with the proof, $\mathbf{Z} = \boldsymbol{\Sigma}^{-\frac{1}{2}} (\mathbf{Y} - \boldsymbol{\mu}) = \boldsymbol{\Sigma}^{-\frac{1}{2}}\mathbf{Y} - \boldsymbol{\Sigma}^{-\frac{1}{2}}\boldsymbol{\mu}$

$$f(\mathbf{Y}) = \frac{1}{(2\pi)^\frac{n}{2}} |\boldsymbol{\Sigma}|^{-\frac{1}{2}}e^{-\frac{1}{2}(\mathbf{Y} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{Y} - \boldsymbol{\mu})}$$

$$\mathbf{Z} = \boldsymbol{\Sigma}^{-\frac{1}{2}} (\mathbf{Y} - \boldsymbol{\mu}) \quad \Rightarrow \quad \mathbf{Y} = \boldsymbol{\Sigma}^\frac{1}{2}\mathbf{Z} + \boldsymbol{\mu}  \qquad \text{Jacobian}: \mathbf{J_2} = |\boldsymbol{\Sigma}^{-\frac{1}{2}}|^{-1} = |\boldsymbol{\Sigma}|^\frac{1}{2}$$

$$f(\mathbf{Z}) = \frac{1}{(2\pi)^\frac{n}{2}} |\boldsymbol{\Sigma}|^{-\frac{1}{2}} \exp\Big(-\frac{1}{2}[(\boldsymbol{\Sigma}^\frac{1}{2}\mathbf{Z} + \boldsymbol{\mu}) - \boldsymbol{\mu}]^T \boldsymbol{\Sigma}^{-1} [(\boldsymbol{\Sigma}^\frac{1}{2}\mathbf{Z} + \boldsymbol{\mu}) - \boldsymbol{\mu}]\Big) \cdot |\boldsymbol{\Sigma}|^{\frac{1}{2}} = \frac{1}{(2\pi)^\frac{n}{2}} e^{-\frac{1}{2}\mathbf{Z}^T\mathbf{Z}}$$

Therefore, $\mathbf{Z} \sim N_n(\mathbf{0}, \mathbf{I})$.

</div>

### Result 3

Suppose $\mathbf{Y} \sim N_n(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then $M_\mathbf{Y}(\mathbf{t}) = e^{\mathbf{t}^T \boldsymbol{\mu} + \frac{1}{2}\mathbf{t}^T \boldsymbol{\Sigma} \mathbf{t}}$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

Let $\mathbf{Z} = \boldsymbol{\Sigma}^{-\frac{1}{2}} (\mathbf{Y} - \boldsymbol{\mu}) \quad \Rightarrow \quad \mathbf{Y} = \boldsymbol{\Sigma}^\frac{1}{2}\mathbf{Z} + \boldsymbol{\mu}$

$$
\begin{aligned}
M_\mathbf{Y}(\mathbf{t}) 
&= M_{\boldsymbol{\Sigma}^\frac{1}{2}\mathbf{Z} + \boldsymbol{\mu}}(\mathbf{t}) \\
&= E[e^{\mathbf{t}^T (\boldsymbol{\Sigma}^\frac{1}{2}\mathbf{Z} + \boldsymbol{\mu}) }] \\
&= e^{\mathbf{t}^T \boldsymbol{\mu}} E[e^{(\boldsymbol{\Sigma}^\frac{1}{2} \mathbf{t})^T \mathbf{Z}}] \\
&= e^{\mathbf{t}^T \boldsymbol{\mu}} E[e^{\mathbf{t}^{*T} \mathbf{Z}}] 
\qquad \text{where } \mathbf{t}^* = \boldsymbol{\Sigma}^\frac{1}{2} \mathbf{t} \\
&= e^{\mathbf{t}^T \boldsymbol{\mu}} M_\mathbf{Z}(\mathbf{t}^*) \\
&= e^{\mathbf{t}^T \boldsymbol{\mu}} e^{\frac{1}{2}\mathbf{t}^{*T}\mathbf{t}^*} \\
&= e^{\mathbf{t}^T \boldsymbol{\mu}} e^{\frac{1}{2}\mathbf{t}^{T}\boldsymbol{\Sigma}^\frac{1}{2}\boldsymbol{\Sigma}^\frac{1}{2} \mathbf{t}} \\
&= e^{\mathbf{t}^T \boldsymbol{\mu}} e^{\frac{1}{2}\mathbf{t}^{T}\boldsymbol{\Sigma} \mathbf{t}}
\end{aligned}
$$

</div>

### Theorem 1

Let $\mathbf{Y} \sim N_n(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, and let $\mathbf{A}$ be an $m \times n$ matrix of rank $m$ and $\mathbf{c}$ be an $m \times 1$ vector. Then $\mathbf{A}\mathbf{Y} + \mathbf{c} \sim N_m(\mathbf{A}\boldsymbol{\mu} + \mathbf{c}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$.

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

$$
\begin{aligned}
M_{\mathbf{A}\mathbf{Y} + \mathbf{c}}(\mathbf{t})
&= E[e^{\mathbf{t}^T (\mathbf{A}\mathbf{Y} + \mathbf{c})}] \\
&= e^{\mathbf{t}^T \mathbf{c}} E[e^{(\mathbf{A}^T\mathbf{t})^T \mathbf{Y}}] \\
&= e^{\mathbf{t}^T \mathbf{c}} E[e^{\mathbf{t}^{*T} \mathbf{Y}}] \qquad \text{where } \mathbf{t}^* = \mathbf{A}^T\mathbf{t} \\
&= e^{\mathbf{t}^T \mathbf{c}} M_\mathbf{Y}(\mathbf{t}^*) \\
&= e^{\mathbf{t}^T \mathbf{c}} e^{\mathbf{t}^{*T}\boldsymbol{\mu} + \frac{1}{2}\mathbf{t}^{*T} \boldsymbol{\Sigma} \mathbf{t}^*} \\
&= e^{\mathbf{t}^T \mathbf{c}} e^{\mathbf{t}^T\mathbf{A}\boldsymbol{\mu} + \frac{1}{2}\mathbf{t}^T\mathbf{A} \boldsymbol{\Sigma} \mathbf{A}^T\mathbf{t}} \\
&= e^{\mathbf{t}^T (\mathbf{A}\boldsymbol{\mu} + \mathbf{c}) + \frac{1}{2}\mathbf{t}^T\mathbf{A} \boldsymbol{\Sigma} \mathbf{A}^T\mathbf{t}} 
\end{aligned}
$$

Therefore, $\mathbf{A}\mathbf{Y} + \mathbf{c} \sim N_m(\mathbf{A}\boldsymbol{\mu} + \mathbf{c}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$

</div>

### Theorem 2

Let $\mathbf{Y} \sim N_n(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. Sub-vectors of $\mathbf{Y}$ follow the multivariate normal distribution and linear combinations of $Y_1, Y_2, \dots, Y_n$ follow the univariate normal distribution

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

Suppose $\mathbf{Y}$, $\boldsymbol{\mu}$, and $\boldsymbol{\Sigma}$ are partitioned as $\mathbf{Y} = \begin{pmatrix} \mathbf{Q_1} \\ \mathbf{Q_2} \end{pmatrix}$, $\boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}$, $\boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix}$. where $\mathbf{Y}_1$ is $p \times 1$. The result follows directly by using the previous theorem with $\mathbf{A} = (\mathbf{I}_p, \mathbf{0})$. For a linear combination $\mathbf{a}^T\mathbf{Y} = a_1Y_1 + a_2Y_2 + \dots + a_nY_n$, $\mathbf{A}$ of theorem 1 is a vector and therefore $\mathbf{a}^T\mathbf{Y} \sim N(\mathbf{a}^T\boldsymbol{\mu}, \mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a})$.

</div>

#### Example

Let $$
\mathbf{Y} = \begin{pmatrix} Y_1 \\ Y_2 \\ \hline Y_3 \\ Y_4 \\ Y_5 \end{pmatrix}, \quad
\boldsymbol{\mu} = \begin{pmatrix} \mu_1 \\ \mu_2 \\ \hline \mu_3 \\ \mu_4 \\ \mu_5 \end{pmatrix}, \quad
\boldsymbol{\Sigma} =
\left(
\begin{array}{cc|ccc}
\sigma_1^2   & \sigma_{12} & \sigma_{13} & \sigma_{14} & \sigma_{15} \\
\sigma_{21}  & \sigma_2^2  & \sigma_{23} & \sigma_{24} & \sigma_{25} \\
\hline
\sigma_{31}  & \sigma_{32} & \sigma_3^2  & \sigma_{34} & \sigma_{35} \\
\sigma_{41}  & \sigma_{42} & \sigma_{43} & \sigma_4^2  & \sigma_{45} \\
\sigma_{51}  & \sigma_{52} & \sigma_{53} & \sigma_{54} & \sigma_5^2
\end{array}
\right)
$$

then if $\mathbf{Q_1} = \begin{pmatrix} Y_1 \\ Y_2 \end{pmatrix}$, it follows that $\mathbf{Q_1} \sim N\Big[ \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, \begin{pmatrix} \sigma_1^2 & \sigma_{12} \\ \sigma_{21} & \sigma_2^2 \end{pmatrix}\Big]$

Alternatively, let $\mathbf{A} = \begin{pmatrix} \mathbf{I}_2 & \mathbf{0} \end{pmatrix}$ then 

$$\mathbf{A}\mathbf{Y} = \mathbf{Q_1} \sim N_p(\mathbf{A}\boldsymbol{\mu}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T) = N_p\Big[ \begin{pmatrix} \mathbf{I}_2 & \mathbf{0} \end{pmatrix} \begin{pmatrix} \boldsymbol{\mu_1} \\ \boldsymbol{\mu_2} \end{pmatrix}, \begin{pmatrix} \mathbf{I}_2 & \mathbf{0} \end{pmatrix} \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix} \begin{pmatrix} \mathbf{I}_2 \\ \mathbf{0} \end{pmatrix} \Big] = \begin{pmatrix} \boldsymbol{\mu_1} & \boldsymbol{\Sigma}_{11} \end{pmatrix}$$

## Statistical independence

Suppose $\mathbf{Y}$, $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$ are partitioned as in Theorem 2. We say $\boldsymbol{Q_1}$ and $\boldsymbol{Q_2}$ are statistically independent if and only if $\boldsymbol{\Sigma}_{12} = \mathbf{0}$. We can show this using the joint moment generating function of $\mathbf{Y}$. Recall that the exponent of the joint moment generating function of the multivariate normal distribution is $\mathbf{t}^T\boldsymbol{\mu} + \frac{1}{2}\mathbf{t}^T\boldsymbol{\Sigma}\mathbf{t}$ which after partitioning $\mathbf{t}$ conformably can be expressed as $\mathbf{t_1}^T\boldsymbol{\mu}_1 + \mathbf{t_2}^T\boldsymbol{\mu}_2 + \frac{1}{2}\mathbf{t_1}^T\boldsymbol{\Sigma}_{11}\mathbf{t_1} + \frac{1}{2}\mathbf{t_2}^T\boldsymbol{\Sigma}_{22}\mathbf{t_2} + \mathbf{t_1}^T\boldsymbol{\Sigma}_{12}\mathbf{t_2}$. When $\boldsymbol{\Sigma}_{12} = \mathbf{0}$, the joint moment generating function can be expressed as the product of the two marginal moment generating functions of $\mathbf{Q_1}$ and $\mathbf{Q_2}$, i.e. $M_\mathbf{Y}(\mathbf{t}) = M_\mathbf{Q_1}(\mathbf{t_1})M_\mathbf{Q_2}(\mathbf{t_2})$. Therefore, $\mathbf{Q_1}$ and $\mathbf{Q_2}$ are independent.

### Theorem 3

Suppose $\mathbf{Y} \sim N_n(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and define the following two vectors: $\mathbf{W_1} = \mathbf{A}\mathbf{Y}$ and $\mathbf{W_2} = \mathbf{B}\mathbf{Y}$. Then, $\mathbf{W_1}$ and $\mathbf{W_2}$ are independent if $cov[\mathbf{W_1},\mathbf{W_2}] = \mathbf{A}\boldsymbol{\Sigma}\mathbf{B}^T = 0$.

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

We stack the two vectors as follows: $\mathbf{W} = \begin{pmatrix} \mathbf{W_1} \\ \mathbf{W_2} \end{pmatrix} = \begin{pmatrix} \mathbf{A} \\ \mathbf{B} \end{pmatrix} \mathbf{Y} = \mathbf{L}\mathbf{Y}$. Therefore using Theorem 1, we find that $\mathbf{W} \sim N(\mathbf{L}\boldsymbol{\mu}, \mathbf{L}\boldsymbol{\Sigma}\mathbf{L}^T)$ or $\mathbf{W} \sim N\Big[ \begin{pmatrix} \mathbf{A} \\ \mathbf{B} \end{pmatrix} \boldsymbol{\mu}, \begin{pmatrix} \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T & \mathbf{A}\boldsymbol{\Sigma}\mathbf{B}^T \\ \mathbf{B}\boldsymbol{\Sigma}\mathbf{A}^T & \mathbf{B}\boldsymbol{\Sigma}\mathbf{B}^T \end{pmatrix}\Big]$, and we conclude that $\mathbf{W_1}$ and $\mathbf{W_2}$ are independent if and only if $\mathbf{A}\boldsymbol{\Sigma}\mathbf{B}^T = \mathbf{0}$

</div>

## Conditional probability density functions for multivariate normal distribution

Consider the bivariate normal distribution. From Theorem 1, it follows that $Y_1 \sim N(\mu_1, \sigma_1)$. This is called the marginal probability distribution of $Y_1$. From the conditional probability law, $f_{Y_2 \mid Y_1}(y_2 \mid y_1) = \frac{f_{Y_1,Y_2}(y_1,y_2)}{f_{Y_1}(y_1)}$, and after substituting the bivariate density and the marginal density, it can be shown that the conditional probability density function of $Y_2$ given $Y_1$ is

$$f_{Y_2 \mid Y_1}(y_2 \mid y_1) = \frac{1}{\sqrt{2\pi} \sqrt{\sigma_2^2(1-\rho^2)}} \exp\Big[ -\frac{1}{2} \frac{\Big( Y_2 - \mu_2 - \rho\frac{\sigma_2}{\sigma_1}(Y_1 - \mu_1) \Big)^2}{\sigma_2^2(1-\rho^2)} \Big]$$

We can recognize that this is a normal probability density function with mean $\mu_{Y_2 \mid Y_1} = \mu_2 + \rho\frac{\sigma_2}{\sigma_1}(Y_1 - \mu_1)$ and variance $\sigma^2_{Y_2 \mid Y_1} = \sigma^2_2(1-\rho^2)$.

### Theorem 4

Suppose Suppose $\mathbf{Y}$, $\boldsymbol{\mu}$, and $\boldsymbol{\Sigma}$ are partitioned as $\mathbf{Y} = \begin{pmatrix} \mathbf{Q_1} \\ \mathbf{Q_2} \end{pmatrix}$, $\boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}$, $\boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix}$, and $\mathbf{Y} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then $\mathbf{Q_1} \mid \mathbf{Q_2} \sim N(\boldsymbol{\mu}_{1 \mid 2}, \boldsymbol{\Sigma}_{1 \mid 2})$, where $\boldsymbol{\mu}_{1 \mid 2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{Q_2} - \boldsymbol{\mu}_2)$, and $\boldsymbol{\Sigma}_{1 \mid 2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$.

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

Let $\mathbf{U} = \mathbf{Q_1} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\mathbf{Q_2}$ and $\mathbf{V} = \mathbf{Q_2}$, then

$$\begin{pmatrix} \mathbf{U} \\ \mathbf{V} \end{pmatrix} = \begin{pmatrix} \mathbf{I} & - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1} \\ \mathbf{0} & \mathbf{I} \end{pmatrix} \begin{pmatrix} \mathbf{Q_1} \\ \mathbf{Q_2} \end{pmatrix} = \mathbf{A}\mathbf{Y}$$

$$E\begin{pmatrix} \mathbf{U} \\ \mathbf{V} \end{pmatrix} = \begin{pmatrix} \mathbf{I} & - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1} \\ \mathbf{0} & \mathbf{I} \end{pmatrix} \begin{pmatrix} \mathbf{\mu}_1 \\ \mathbf{\mu}_2 \end{pmatrix} = \begin{pmatrix} \mathbf{\mu}_1 - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\mathbf{\mu}_2 \\ \mathbf{\mu}_2 \end{pmatrix}$$

$$
\begin{aligned}
var\begin{pmatrix} \mathbf{U} \\ \mathbf{V} \end{pmatrix} = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^{-1} 
&= \begin{pmatrix} \mathbf{I} & - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1} \\ \mathbf{0} & \mathbf{I} \end{pmatrix} \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix} \begin{pmatrix} \mathbf{I} & \mathbf{0} \\ - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1} & \mathbf{I} \end{pmatrix}
\\
&= \begin{pmatrix} \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21} & \mathbf{0} \\ \mathbf{0} & \boldsymbol{\Sigma}_{22} \end{pmatrix}
\end{aligned}
$$

Thus $\mathbf{U}$ and $\mathbf{V}$ are independent since $cov[\mathbf{U}, \mathbf{V}] = 0$

$$\mathbf{Q_1} \mid \mathbf{Q_2} = \mathbf{U} \mid \mathbf{Q_2} + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\mathbf{Q_2}$$

$$
\begin{aligned}
E[\mathbf{Q_1} \mid \mathbf{Q_2}] 
&= E[\mathbf{U} \mid \mathbf{Q_2} + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\mathbf{Q_2}] \\
&= \boldsymbol{\mu}_1 - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\mu}_2 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\mathbf{Q_2} \\
&= \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{Q_2} - \boldsymbol{\mu}_2) \\
\end{aligned}
$$

$$var[\mathbf{Q_1} \mid \mathbf{Q_2}] = var[\mathbf{U}] = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

</div>


--- 


# Lec 7: Distributions related to the normal distribution

## Gamma distribution

A random variable $X$ is said to have a gamma distribution with parameters $\alpha$, $\beta$ if its probability density function is given by

$$f(x)=\frac{x^{\alpha-1}e^{-\frac{x}{\beta}}}{\beta^\alpha\Gamma(\alpha)}, \quad \alpha,\beta > 0, x \geq 0$$

with $E[X] = \alpha\beta$, $var[X] = \alpha\beta^2$, and $\Gamma(\alpha) = \int_0^\infty x^{\alpha-1}e^{-x} dx$. If set $\alpha = 1$ and $\beta = \frac{1}{\lambda}$, we have $f(x) = \lambda e^{-\lambda x}$. We see that exponential distribution is a special case of gamma distribution.


### Moment generating function - gamma distribution

Let $X \sim \Gamma(\alpha, \beta)$, then $M_X(t) = (1-\beta t)^{-\alpha}$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

$$
\begin{aligned}
M_{X}(t) 
&= E[e^{tX}] \\
&= \int_0^\infty e^{tx} \frac{x^{\alpha-1}e^{-\frac{x}{\beta}}}{\beta^\alpha\Gamma(\alpha)} dx \\
&= \frac{1}{\beta^\alpha\Gamma(\alpha)} \int_0^\infty x^{\alpha-1}e^{-x(\frac{1-\beta t}{\beta})}dx \\
&= \frac{1}{\beta^\alpha\Gamma(\alpha)} \int_0^\infty \Big(\frac{\beta}{1-\beta t}\Big)^{\alpha-1} y^{\alpha-1}e^{-y} \frac{\beta}{1-\beta t} dy \\
\text{where } y &= x(\frac{1-\beta t}{\beta}) \Rightarrow x = (\frac{\beta}{1-\beta t})y, dx = (\frac{\beta}{1-\beta t}) dy \\
M_{X}(t) &= \Big(\frac{\beta}{1-\beta t}\Big)^{\alpha} \int_0^\infty \frac{y^{\alpha-1}e^{-y}}{\beta^\alpha\Gamma(\alpha)} dy \\ 
&= (1 - \beta t)^{-\alpha}
\end{aligned}
$$

</div>

## $\chi^2$ distribution

### Definition

Let $Z \sim N(0,1)$, then $X = Z^2$ where $f(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}$. We say that $X$ follows the Chi-square distribution with 1 degree of freedom denoted $X \sim \chi^2_1$. Find the pdf of $X$ beginning with the cdf of $X$:

$$
\begin{aligned}
F_X(x) &= P(X \leq x) = P(Z^2 \leq x) = P(-\sqrt{x} \leq Z \leq \sqrt{x}) = F_Z(\sqrt{x}) - F_Z(-\sqrt{x}) \\
f_X(x) &= 
\frac{1}{2}x^{-\frac{1}{2}}\frac{1}{\sqrt{2\pi}}e^{-\frac{x}{2}} + 
\frac{1}{2}x^{-\frac{1}{2}}\frac{1}{\sqrt{2\pi}}e^{-\frac{x}{2}} = 
\frac{x^{-\frac{1}{2}}e^{-\frac{x}{2}}}{2^{\frac{1}{2}}\sqrt{\pi}} =
\frac{x^{-\frac{1}{2}}e^{-\frac{x}{2}}}{2^{\frac{1}{2}}\Gamma(\frac{1}{2})}
\end{aligned}
$$

This is the pdf of $\Gamma(\frac{1}{2}, 2)$ or $\chi^2_1$, and the moment generating function is $M_X(t) = (1 - 2t)^{-\frac{1}{2}}$. Since the mean of $\Gamma(\alpha, \beta)$ is $E[X] = \alpha\beta$ and its variance $var[X] = \alpha\beta^2$. Therefore, if $Y \sim \chi^2_n$, it follows that:

$$E[Y] = n \qquad var[Y] = 2n$$

In general, the shape of distribution is skewed to the right, but as the degree of freedom increase, it becomes close to $N(n, 2n)$ (central limit theorem).

### Theorem 

Let $Z_1, Z_2, \dots, Z_n$ be independent random variables with $Z_i \sim N(0,1)$, then $Z_1^2 + Z_2^2 + \dots + Z_n^2 \sim \chi^2_n$ (Chi-square distribution with $n$ degrees of freedom).

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

Since $Z_1, Z_2, \dots, Z_n$ are independent, therefore:

$$M_{\sum_{i=1}^nZ_i^2}(t) = \prod_{i=1}^n M_{Z_i^2}(t) = \prod_{i=1}^n (1 - 2t)^{-\frac{1}{2}} = (1 - 2t)^{-\frac{n}{2}}$$

This is the mgf of $\Gamma(\frac{n}{2}, 2)$ and is called the Chi-square distribution with $n$ degrees of freedom.

</div>

### Theorem

Let $X_1, X_2, \dots, X_n$ be independent random variables with $X_i \sim N(\mu,\sigma)$. It follows from the previous theorem that 

$$\sum_{i=1}^n \Big( \frac{X_i - \mu}{\sigma} \Big)^2 \sim \chi^2_n$$

### Theorem 

Let $X \sim \chi^2_n$ and $Y \sim \chi^2_m$, $X \perp\!\!\!\perp Y$, then it can be shown using moment generating function that

$$X + Y \sim \chi^2_{n+m}$$

### Theorem

Let $X_1, X_2, \dots, X_n$ be independent random variables with $X_i \sim N(\mu, \sigma^2)$, so $\mathbf{X} = \sum_{i=1}^n X_i \sim N_n(\mu\mathbf{1}, \sigma^2\mathbf{I})$. Define the sample variance as 

$$s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2 \qquad \text{Then } \frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}$$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

Begin with $\sum_{i=1}^n (\frac{X_i - \mu}{\sigma})^2 \sim \chi^2_n$.

$$
\begin{aligned}
\frac{\sum_{i=1}^n (X_i - \mu)^2}{\sigma^2} 
&= \frac{\sum_{i=1}^n (X_i - \bar{X} + \bar{X} - \mu)^2}{\sigma^2} \\
&= \frac{\sum_{i=1}^n (X_i - \bar{X})^2}{\sigma^2} + \frac{n(\bar{X} - \mu)^2}{\sigma^2} + \frac{2(\bar{X} - \mu) \overbrace{\sum_{i=1}^n (X_i - \bar{X})}^{\text{equal to } 0} }{\sigma^2} \\
&= \frac{(n-1)s^2}{\sigma^2} + \Big( \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \Big)^2 \\
&= \mathbf{Q_1} + \mathbf{Q_2}
\end{aligned}
$$

Note: $(\mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^T)\mathbf{X} = \begin{pmatrix} (X_1 - \bar{X}) \\ \vdots \\ (X_n - \bar{X}) \end{pmatrix}$ is the mean-centered version of $\mathbf{X}$.

$$
\begin{aligned}
cov[\bar{X}, (\mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^T)\mathbf{X}]
&= cov[\frac{1}{n}\mathbf{1}^T\mathbf{X}, (\mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^T)\mathbf{X}] \qquad &\text{Note: } cov[\mathbf{A}\mathbf{X}, \mathbf{B}\mathbf{X}] = \mathbf{A}\boldsymbol{\Sigma}\mathbf{B} \\
&= \frac{1}{n}\mathbf{1}^T \sigma^2\mathbf{I} (\mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^T)^T \qquad &\text{Note: } (\mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^T) \text{ is symmetric} \\ 
&= \frac{\sigma^2}{n} \mathbf{1}^T (\mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^T) \\
&= \frac{\sigma^2}{n} (\mathbf{1}^T - \frac{1}{n}\mathbf{1}^T\mathbf{1}\mathbf{1}^T) \\
&= \frac{\sigma^2}{n} (\mathbf{1}^T - \mathbf{1}^T) \\
&= \mathbf{0}
\end{aligned}
$$

By normality, $\bar{X} \perp\!\!\!\perp (\mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^T)\mathbf{X} = \mathbf{A}\mathbf{X}$. Since $s^2 = g(\mathbf{A}\mathbf{X}) = \frac{1}{n-1}(\mathbf{A}\mathbf{X})^T(\mathbf{A}\mathbf{X})$, so $s^2$ is a deterministic function of $\mathbf{A}\mathbf{X}$ and therefore $\bar{X}$ and $s^2$ are independent.

$$
M_\mathbf{Q}(t) = M_\mathbf{Q_1}(t) M_\mathbf{Q_2}(t) \quad \Rightarrow \quad 
M_\mathbf{Q_1}(t) = \frac{M_\mathbf{Q}(t)}{M_\mathbf{Q_2}(t)}
$$

$$
\text{Note: } \bar{X} \sim N(\mu, \frac{\sigma^2}{n}) \qquad
\text{so } (\frac{\bar{X} - \mu}{\sigma/\sqrt{n}})^2 \sim \chi^2_1
$$

$$
\text{Thus } M_\mathbf{Q_1}(t) = \frac{(1 - 2t)^{-\frac{n}{2}}}{(1 - 2t)^{-\frac{1}{2}}} = (1 - 2t)^{-\frac{n-1}{2}} \qquad \mathbf{Q_1} = \frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}
$$

</div>

## $t$ distribution

### Definition

Let $Z \sim N(0,1)$ and $U \sim \chi^2_{df}$. If $Z$ and $U$ are independent, then the ratio $X = \frac{Z}{\sqrt{\frac{U}{df}}}$ follows the $t$ (or Student's $t$) distribution with degrees of freedom equal to $df$. We write $X \sim t_{df}$. The pdf of the $t$ distribution with $df = n$ degrees of freedom is 

$$f(x) = \frac{\Gamma(\frac{n+1}{2})}{\sqrt{\pi n}\Gamma(\frac{n}{2})} \Big( 1 + \frac{x^2}{n} \Big)^{-\frac{n+1}{2}}, \qquad -\infty < x < \infty$$

Let $X \sim t_n$. Then, $E[X] = 0$ and $var[X] = \frac{n}{n-2}$. The $t$ distribution is similar to the standard normal distribution $N(0,1)$, but it has heavier tails. However as $n \rightarrow \infty$ the $t$ distribution coverges to $N(0,1)$.

### Application

Let $X_1, X_2, \dots, X_n$ be independent and indentically distributed random variables each one having $N(\mu, \sigma^2)$. Construct a $t$ distribution using the definition of the $t$ distribution.

$$\frac{\bar{X}-\mu}{s/\sqrt{n}} = \frac{(\bar{X}-\mu) / (\sigma/\sqrt{n})}{\sqrt{\frac{(n-1)s^2}{\sigma^2} / (n-1)}} \sim t_{n-1}$$

### Example 

Let $\bar{X}$ and $s_X^2$ denote the sample mean and sample variance of an independent random sample of size 10 from a normal distribution with mean $\mu=0$ and variance $\sigma^2$. Find $c$ so that 

$$P\Big( \frac{\bar{X}}{\sqrt{9s_X^2}} < c \Big) = 0.95$$

Solution: 

$$
\begin{aligned}
&P\Big( \frac{\bar{X}}{\sqrt{9s_X^2}} < c \Big) = 0.95 \\
\quad \Rightarrow \quad &P\Big( \frac{(\bar{X}-0) / (\sigma/\sqrt{10})}{\sqrt{\frac{(10-1)s_X^2}{\sigma^2} / (10-1)}} < \sqrt{90} c \Big) = 0.95 \\
\quad \Rightarrow \quad &P\Big( t_{9} < \sqrt{90} c \Big) = 0.95 \\
\quad \Rightarrow \quad &\sqrt{90} c = t_{0.95,9} = 1.833 \\
\quad \Rightarrow \quad &c = 0.193
\end{aligned}
$$

## $F$ distribution

### Definition

Let $U \sim \chi^2_{n_1}$ and $V \sim \chi^2_{n_2}$. If $U$ and $V$ are independent, the ratio $X = \frac{U/n_1}{V/n_2}$ follows the $F$ distribution with numerator $df = n_1$ and denominator $df = n_2$. We write $X \sim F_{n_1,n_2}$. The pdf of $X$ is

$$f(x) = \frac{\Gamma(\frac{n_1+n_2}{2})}{\Gamma(\frac{n_1}{2})\Gamma(\frac{n_2}{2})} \Big( \frac{n_1}{n_2} \Big)^{\frac{n_1}{2}} x^{\frac{n_1}{2}-1} \Big( 1+\frac{n_1}{n_2}x \Big)^{-\frac{1}{2}(n_1+n_2)}, \quad 0 < x < \infty$$

Let $X \sim F_{n_1,n_2}$. Then $E[X] = \frac{n_2}{n_2-2}$ and $var[X] = \frac{2n_2^2(n_1+n_2-2)}{n_1(n_2-2)^2(n_2-4)}$. In general, the $F$ distribution is skewed to the right.

**Properties**:
- If $X \sim F_{n,m}$, then $\frac{1}{X} \sim F_{m,n}$
- $F_{\alpha; m,n} = \frac{1}{1-\alpha; n,m}$
- $F_{1,n} = t_n^2$

### Application

Let $X_1, X_2, \dots, X_n \sim \overset{\text{i.i.d.}}{\sim} N(\mu_X, \sigma_X^2)$.

Let $Y_1, Y_2, \dots, Y_m \sim \overset{\text{i.i.d.}}{\sim} N(\mu_Y, \sigma_Y^2)$.

The two samples are independent. Use $s_X^2$, $s_Y^2$ and $\sigma_X^2$, $\sigma_Y^2$ to form a ratio that follows $F_{n-1,m-1}$

$$
\frac{(n-1)s_X^2}{\sigma_X^2} \sim \chi^2_{n-1} \qquad
\frac{(m-1)s_Y^2}{\sigma_Y^2} \sim \chi^2_{m-1}
$$

$$\frac{\frac{(n-1)s_X^2}{\sigma_X^2} / (n-1)}{\frac{(m-1)s_Y^2}{\sigma_Y^2} / (m-1)} = \frac{s_X^2 / \sigma_X^2}{s_Y^2 / \sigma_Y^2} \sim F_{n-1,m-1}$$

### Example 

Two independent samples of size $n_1 = 6$, $n_2 = 10$ are taken from two normal populations with equal variances. Find $b$ such that $P(\frac{s_1^2}{s_2^2} < b) = 0.95$

$$
\frac{(6-1)s_1^2}{\sigma^2} \sim \chi^2_{6-1} \qquad
\frac{(10-1)s_2^2}{\sigma^2} \sim \chi^2_{10-1}
$$

$$\frac{\frac{(6-1)s_1^2}{\sigma^2} / (6-1)}{\frac{(10-1)s_2^2}{\sigma^2} / (10-1)} = \frac{s_1^2}{s_2^2} \sim F_{5,9}$$

$$P\Big( \frac{s_1^2}{s_2^2} < b \Big) = 0.95 \quad \Rightarrow \quad b = F_{0.95;5,9} = 3.482$$

## Non-central distribution

### Non-central $\chi^2$ distribution

Let $Y \sim N(\mu, 1)$, then $Y^2 \sim \chi^2_1(ncp=\mu^2)$ (ncp: non-centrality parameter). If $Y \sim N(\mu, \sigma^2)$, then $(\frac{Y}{\sigma})^2 \sim \chi^2_1(ncp=\frac{\mu^2}{\sigma^2})$.

**Moment generating function**: Let $Y \sim \chi^2_1(ncp=\theta)$, then $M_Y(t) = (1 - 2t)^{-\frac{1}{2}} e^{\theta \frac{t}{1-2t}}$

**Example**: Let $Y_1,\dots,Y_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$, find the distribution of $\sum_{i=1}^n \frac{Y_i^2}{\sigma^2}$ using MGF: 

$$M_{\sum_{i=1}^n \frac{Y_i^2}{\sigma^2}}(t) = \prod_{i=1}^n M_{\frac{Y_i^2}{\sigma^2}}(t) = \prod_{i=1}^n (1 - 2t)^{-\frac{1}{2}} e^{\frac{\mu^2}{\sigma^2} \frac{t}{1-2t}} = (1 - 2t)^{-\frac{n}{2}} e^{\frac{n \mu^2}{\sigma^2} \frac{t}{1-2t}}$$

Therefore, $\sum_{i=1}^n \frac{Y_i^2}{\sigma^2} \sim \chi^2_n(ncp = \frac{n \mu^2}{\sigma^2})$

### Non-central $t$ distribution

Let $Z \sim N(\delta, 1)$, $U \sim \chi^2_{df}$, and $Z \perp\!\!\!\perp U$, then $\frac{Z}{\sqrt{U/df}} \sim t_{df}(ncp=\delta)$

### Non-central $F$ distribution

Let $U \sim \chi^2_{n_1}(ncp=\theta)$, $V \sim \chi^2_{n_2}(ncp=\theta)$, and $U \perp\!\!\!\perp V$, then $\frac{U/n_1}{V/n_2} \sim F_{n_1,n_2}(ncp=\theta)$


---


# Lec 8: Properties of estimators

## Unbiased estimators

### Definition

Let $\hat{\theta}$ be an estimator of a parameter $\theta$. We say that $\hat{\theta}$ is an unbiased estimator of $\theta$ if $E[\hat{\theta}] = \theta$.

### Example

Let $X_1, X_2, \dots, X_n$ be an i.i.d. sample from a population with mean $\mu$ and variance $\sigma^2$. Show that $\bar{X}$ and $s^2$ are unbiased estimators of $\mu$ and $\sigma^2$ respectively.

$$E[\bar{X}] = E\Big[ \frac{1}{n} \sum_{i=1}^n X_i \Big] = \frac{1}{n} \sum_{i=1}^n E[X_i] = \frac{1}{n} n\mu = \mu$$

$$
\begin{aligned}
E[s^2] = E\Big[ \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2 \Big] 
&= \frac{1}{n-1} E\Big[ \sum_{i=1}^n ((X_i - \mu) - (\bar{X} - \mu))^2 \Big] \\
&= \frac{1}{n-1} E\Big[ (\sum_{i=1}^n (X_i - \mu)^2) + n(\bar{X} - \mu)^2 - 2(\bar{X} - \mu)\sum_{i=1}^n(X_i - \mu) \Big] \\
&= \frac{1}{n-1} E\Big[ (\sum_{i=1}^n (X_i - \mu)^2) + n(\bar{X} - \mu)^2 - 2(\bar{X} - \mu) \cdot n(\bar{X} - \mu) \Big] \\
&= \frac{1}{n-1} E\Big[ (\sum_{i=1}^n (X_i - \mu)^2) - n(\bar{X} - \mu)^2 \Big] \\
&= \frac{1}{n-1} ((\sum_{i=1}^n var[X]) - n(var[\bar{X}])) \\
&= \frac{1}{n-1} E\Big[ n\sigma^2 - \sigma^2 \Big] \\
&= \sigma^2
\end{aligned}
$$

### Example 

Let $X \sim bin(n,p)$. Show that $\hat{p} = \frac{X}{n}$ is an unbiased estimator of $p$

$$E[\hat{p}] = E[\frac{X}{n}] = \frac{1}{n}E[X] = \frac{1}{n} \cdot np = p$$

## Information and Cramér-Rao inequality

### Information 

Let $X$ be a random variable with pdf $f(x;\theta)$, then 

$$
\begin{aligned}
&\int_{-\infty}^\infty f(x;\theta) dx = 1 \qquad \text{take derivative w.r.t. } \theta \text{ on both sides} \\
\Rightarrow \quad &\int_{-\infty}^\infty \frac{\partial f(x;\theta)}{\partial \theta} dx = 0 \\
\Rightarrow \quad &\int_{-\infty}^\infty \frac{1}{f(x;\theta)} \frac{\partial f(x;\theta)}{\partial \theta} f(x;\theta) dx = 0 \\
\Rightarrow \quad &\int_{-\infty}^\infty \frac{\partial \log f(x;\theta)}{\partial \theta} f(x;\theta) dx = 0 \qquad \text{differentiate again w.r.t. } \theta \text{ on both sides} \\
\Rightarrow \quad &\int_{-\infty}^\infty \Big[ \frac{\partial^2 \log f(x;\theta)}{\partial \theta^2} f(x;\theta) + \frac{\partial \log f(x;\theta)}{\partial \theta} \frac{\partial f(x;\theta)}{\partial \theta} \Big] dx = 0 \\
\Rightarrow \quad &\int_{-\infty}^\infty \Big[ \frac{\partial^2 \log f(x;\theta)}{\partial \theta^2} f(x;\theta) + \frac{\partial \log f(x;\theta)}{\partial \theta} \frac{1}{f(x;\theta)} \frac{\partial f(x;\theta)}{\partial \theta} f(x;\theta) \Big] dx = 0 \\
\Rightarrow \quad &\int_{-\infty}^\infty \Big[ \frac{\partial^2 \log f(x;\theta)}{\partial \theta^2} f(x;\theta) + \Big( \frac{\partial \log f(x;\theta)}{\partial \theta} \Big)^2  f(x;\theta) \Big] dx = 0 \\
\Rightarrow \quad &\int_{-\infty}^\infty \frac{\partial^2 \log f(x;\theta)}{\partial \theta^2} f(x;\theta) dx + \int_{-\infty}^\infty \Big( \frac{\partial \log f(x;\theta)}{\partial \theta} \Big)^2  f(x;\theta) dx = 0 \\
\Rightarrow \quad &E\Big[ \frac{\partial^2 \log f(X;\theta)}{\partial \theta^2} \Big] + E\Big[ \Big( \frac{\partial \log f(X;\theta)}{\partial \theta} \Big)^2 \Big] = 0\\
\Rightarrow \quad &E\Big[ \Big( \frac{\partial \log f(X;\theta)}{\partial \theta} \Big)^2 \Big] = -E\Big[ \frac{\partial^2 \log f(X;\theta)}{\partial \theta^2} \Big]
\end{aligned}
$$

The expression

$$I(\theta) = E\Big[ \Big( \frac{\partial \log f(X;\theta)}{\partial \theta} \Big)^2 \Big] = -E\Big[ \frac{\partial^2 \log f(X;\theta)}{\partial \theta^2} \Big]$$

is the information for one observation. The information can also be computed using the variance of the score function $S = \frac{\partial \log f(x;\theta)}{\partial \theta}$:

$$var[S] = E[S^2] - E[S]^2 = E[S^2] = E\Big[ \Big( \frac{\partial \log f(X;\theta)}{\partial \theta} \Big)^2 \Big]$$

### Information in a sample

Let $X_1, X_2, \dots, X_n$ be an i.i.d. random sample from a distribution with pdf $f(x;\theta)$. The joint pdf of $X_1, X_2, \dots, X_n$ is

$$L(\theta) = f(x_1;\theta)f(x_2;\theta) \dots f(x_n;\theta)$$

Take logarithm on both sides

$$\log L(\theta) = \log f(x_1;\theta) + \log f(x_2;\theta) + \dots + \log f(x_n;\theta)$$

Take derivative w.r.t. $\theta$ on both sides

$$\frac{\partial \log L(\theta)}{\partial \theta} = \frac{\partial \log f(x_1;\theta)}{\partial \theta} + \frac{\partial \log f(x_2;\theta)}{\partial \theta} + \dots + \frac{\partial \log f(x_n;\theta)}{\partial \theta}$$

When one observation was involved the information was $E[(\frac{\partial \log f(X;\theta)}{\partial \theta})^2]$. Now, a random sample $X_1, X_2, \dots, X_n$ has $f(x;\theta)$ being replaced by $L(\theta)$. Therefore, the information in the sample will be $E[(\frac{\partial \log L(\theta)}{\partial \theta})^2]$.

$$
\begin{aligned}
\Big( \frac{\partial \log L(\theta)}{\partial \theta} \Big)^2
&= \Big( \frac{\partial \log f(x_1;\theta)}{\partial \theta} + \frac{\partial \log f(x_2;\theta)}{\partial \theta} + \dots + \frac{\partial \log f(x_n;\theta)}{\partial \theta} \Big)^2 \\
&= \Big( \frac{\partial \log f(x_1;\theta)}{\partial \theta} \Big)^2 + \Big( \frac{\partial \log f(x_2;\theta)}{\partial \theta} \Big)^2 + \dots + \Big( \frac{\partial \log f(x_n;\theta)}{\partial \theta} \Big)^2 \\
&+ 2 \frac{\partial \log f(x_1;\theta)}{\partial \theta}\frac{\partial \log f(x_2;\theta)}{\partial \theta} + \dots \\
E\Big[ \Big( \frac{\partial \log L(\theta)}{\partial \theta} \Big)^2 \Big] 
&= E\Big[ \Big( \frac{\partial \log f(X_1;\theta)}{\partial \theta} \Big)^2 \Big] + E\Big[ \Big( \frac{\partial \log f(X_2;\theta)}{\partial \theta} \Big)^2 \Big] + \dots + E\Big[ \Big( \frac{\partial \log f(X_n;\theta)}{\partial \theta} \Big)^2 \Big] 
\end{aligned}
$$

Expectation of cross-product terms equal to zero because 

$$E[\frac{\partial \log f(X_i;\theta)}{\partial \theta}\frac{\partial \log f(X_j;\theta)}{\partial \theta}] = E[\frac{\partial \log f(X_i;\theta)}{\partial \theta}] \cdot E[\frac{\partial \log f(X_j;\theta)}{\partial \theta}] = 0 \cdot 0 = 0$$

We conclude that the information in the sample is equal to $n$ times the information for one observation

$$I_n(\theta) = E\Big[ \Big( \frac{\partial \log L(\theta)}{\partial \theta} \Big)^2 \Big] = I(\theta) + I(\theta) + \dots + I(\theta) = nI(\theta)$$

### Cramér-Rao inequality

Let $X_1, X_2, \dots, X_n$ be an i.i.d. sample from a distribution that has pdf $f(x)$. Let $\hat{\theta}$ be an unbiased estimator of a parameter $\theta$ of this distribution, then the variance of $\theta$ is at least

$$
var[\hat{\theta}] \geq \frac{1}{nE[(\frac{\partial \log f(X;\theta)}{\partial \theta})^2]} = \frac{1}{nI(\theta)} 
\qquad \text{OR} \qquad 
var[\hat{\theta}] \geq \frac{1}{-nE[\frac{\partial^2 \log f(X;\theta)}{\partial \theta^2}]} = \frac{1}{nI(\theta)} 
$$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

Let $X_1, X_2, \dots, X_n$ be an i.i.d. random sample from a distribution with pdf $f(x;\theta)$, and let $\hat{\theta} = g(X_1, X_2, \dots, X_n)$ be an unbiased estimator of the unknown parameter $\theta$. Since $\hat{\theta}$ is unbiased, then

$$E[\hat{\theta}] = \int_{-\infty}^\infty \dots\dots \int_{-\infty}^\infty g(x_1, x_2, \dots, x_n) f(x_1;\theta) f(x_2;\theta) \dots f(x_n;\theta) dx_1 dx_2 \dots dx_n = \theta$$

Take derivatives w.r.t. $\theta$ on both sides

$$
\begin{aligned}
&\int_{-\infty}^\infty \dots\dots \int_{-\infty}^\infty g(x_1, x_2, \dots, x_n) \Big[ \sum_{i=1}^n \frac{1}{f(x_i;\theta)} \frac{\partial f(x_i;\theta)}{\partial \theta} \Big] f(x_1;\theta) f(x_2;\theta) \dots f(x_n;\theta) dx_1 dx_2 \dots dx_n = 1 \\
\Rightarrow &\int_{-\infty}^\infty \dots\dots \int_{-\infty}^\infty g(x_1, x_2, \dots, x_n) \Big[ \sum_{i=1}^n \frac{\partial \log f(x_i;\theta)}{\partial \theta} \Big] f(x_1;\theta) f(x_2;\theta) \dots f(x_n;\theta) dx_1 dx_2 \dots dx_n = 1 \\
\Rightarrow &\int_{-\infty}^\infty \dots\dots \int_{-\infty}^\infty g(x_1, x_2, \dots, x_n) Q f(x_1;\theta) f(x_2;\theta) \dots f(x_n;\theta) dx_1 dx_2 \dots dx_n = 1 \qquad \text{where } Q = \sum_{i=1}^n \frac{\partial \log f(x_i;\theta)}{\partial \theta} \\
\Rightarrow &E[\hat{\theta}Q] = 1
\end{aligned}
$$

Now find the correlation between $\hat{\theta}$ and $Q$

$$
\begin{aligned}
\rho_{\hat{\theta}, Q} \in [-1, 1] \quad &\Rightarrow \quad \rho_{\hat{\theta}, Q}^2 \leq 1 \\
&\Rightarrow \quad \frac{cov[\hat{\theta}, Q]^2}{var[\hat{\theta}] var[Q]} \leq 1 \\
&\Rightarrow \quad \frac{(E[\hat{\theta}Q] - E[\hat{\theta}]E[Q])^2}{var[\hat{\theta}] (E[Q^2] - E[Q]^2)} \leq 1 \\
&\Rightarrow \quad \frac{1 - 0}{var[\hat{\theta}] (nI(\theta) - 0)} \leq 1 \qquad \text{Square function}: E[Q] = 0; E[Q^2] = nI(\theta) \\
&\Rightarrow \quad var[\hat{\theta}] \geq \frac{1}{nI(\theta)}
\end{aligned}
$$

</div>

### Information inequality (generalized Cramér-Rao inequality)

Let $\tau(\boldsymbol{\theta})$ be a function of $\boldsymbol{\theta} = \begin{pmatrix} \theta_1 & \dots & \theta_p \end{pmatrix}^T$. Let $T(x)$ be an unbiased estimator of $\tau(\boldsymbol{\theta})$. if

$$var[T(x)] \geq \nabla \tau^T(\boldsymbol{\theta}) I^{-1}(\boldsymbol{\theta}) \tau(\boldsymbol{\theta}) \qquad \text{where } \nabla \tau(\boldsymbol{\theta}) = \begin{pmatrix} \frac{\partial \tau(\boldsymbol{\theta})}{\partial \theta_1} & \dots & \frac{\partial \tau(\boldsymbol{\theta})}{\partial \theta_p} \end{pmatrix}^T$$

if equal, $T(x)$ is an efficient estimator of $\tau(\boldsymbol{\theta})$

### Example 

Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$. Let $\tau(\mu, \sigma^2) = \mu\sigma^2$, then $T(x) = \bar{X}s^2$

$$
\begin{aligned}
var[T(x)] &\geq 
\begin{pmatrix} \sigma^2 & \mu \end{pmatrix}
\begin{pmatrix} \frac{n}{\sigma^2} & 0 \\ 0 & \frac{n}{2\sigma^4} \end{pmatrix}
\begin{pmatrix} \sigma^2 \\ \mu \end{pmatrix}
= \frac{\sigma^6}{n} + \frac{2\mu^2\sigma^4}{n} \\
var[T(x)] &= E[\bar{X}^2S^4] - E[\bar{X}s^2]^2 \qquad \text{Note: } s^2 \sim \Gamma(\frac{n-1}{n}, \frac{2\sigma^2}{n-1}) \\
&= E[\bar{X}^2] E[S^4] - (\mu\sigma^2)^2 \\
&= \Big( var[\bar{X}] + E[\bar{X}]^2 \Big)  \Big( \frac{\Gamma(\frac{n-1}{2} + 2)(\frac{2\sigma^2}{n-1})^2}{\Gamma(\frac{n-1}{2})} \Big) - \mu^2\sigma^4 \\
&= \Big( \frac{\sigma^2}{n} + \mu^2 \Big)  \Big( \frac{\Gamma(\frac{n-1}{2} + 2)(\frac{2\sigma^2}{n-1})^2}{\Gamma(\frac{n-1}{2})} \Big) - \mu^2\sigma^4
\end{aligned}
$$

### Information matrix

For a p-dimensional parameter vector $\theta$, the Fisher information matrix is given as:

$$
I(\boldsymbol{\theta}) = -E
\begin{pmatrix}
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \theta_1^2} &
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \theta_1 \partial \theta_2} &
\cdots &
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \theta_1 \partial \theta_p} \\
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \theta_2 \partial \theta_1} &
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \theta_2^2} &
\cdots &
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \theta_2 \partial \theta_p} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \theta_p \partial \theta_1} &
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \theta_p \partial \theta_2} &
\cdots &
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \theta_p^2}
\end{pmatrix}
$$

### Example 

Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$.

$$
I(\boldsymbol{\theta}) = -E
\begin{pmatrix}
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \mu^2} &
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \mu \partial \sigma^2} \\
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \sigma^2 \partial \mu} &
\frac{\partial^2 \log L(\boldsymbol{\theta})}{\partial \sigma^4}
\end{pmatrix} = -
\begin{pmatrix}
-\frac{n}{\sigma^2} & 0 \\ 0 & -\frac{n}{2\sigma^4}
\end{pmatrix} \quad \Rightarrow \quad
I^{-1}(\boldsymbol{\theta}) = 
\begin{pmatrix}
\frac{\sigma^2}{n} & 0 \\ 0 & \frac{2\sigma^4}{n}
\end{pmatrix}
$$

On the other hand,

$$
var\Big[ \bar{X} \\ s^2 \Big] = 
\begin{pmatrix}
\frac{\sigma^2}{n} & 0 \\ 0 & \frac{2\sigma^4}{n-1}
\end{pmatrix}
$$

Since $\frac{\sigma^2}{n} \rightarrow \frac{\sigma^2}{n}$ and $\frac{2\sigma^4}{n} \rightarrow \frac{2\sigma^4}{n-1}$ as $n \rightarrow \infty$, then $\bar{X}$ is an efficient estimator and $s^2$ is an asymptotically efficient estimator.

## Efficient estimators

### Definition

We say that $\hat{\theta}$ is an efficient estimator of $\theta$ if $\hat{\theta}$ is an unbiased estimator and 

$$var[\hat{\theta}] = \frac{1}{nI(\theta)}$$

In other words, if the $\hat{\theta}$ attains the minimum variance of the Cramér-Rao inequality.

### Example 

Let $X_1, X_2, \dots, X_n$ be i.i.d. sample from a normal population with mean $\mu$ and variance $\sigma^2$. Show that the information in one observation can be obtained using $I(\theta) = E[(\frac{\partial \log f(x)}{\partial \theta})^2] = -E[\frac{\partial^2 \log f(x)}{\partial \theta^2}] = var[S]$, where $S = \frac{\partial \log f(x)}{\partial \theta}$ is the score function.

$$
\begin{aligned}
f(x) &= (2\pi\sigma^2)^{-\frac{1}{2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \\
\log f(x) &= -\frac{1}{2} \log(2\pi\sigma^2) -\frac{(x-\mu)^2}{2\sigma^2} \\
S_\mu &= \frac{\partial \log f(x)}{\partial \mu} = \frac{2}{2\sigma^2}(X-\mu) = \frac{X-\mu}{\sigma^2}\\
I(\mu) &= var[S^2] = var\Big[ \frac{X-\mu}{\sigma^2} \Big] = \frac{1}{\sigma^4}var[X-\mu] = \frac{var[X]}{\sigma^4} = \frac{1}{\sigma^2} \\
I(\mu) &= -E\Big[ \frac{\partial^2 \log f(x)}{\partial \theta^2} \Big] = -E\Big[ -\frac{1}{\sigma^2} \Big] = \frac{1}{\sigma^2} \\
I(\mu) &= E\Big[ \Big( \frac{\partial \log f(x)}{\partial \theta} \Big)^2 \Big] = E\Big[ \Big( \frac{X-\mu}{\sigma^2} \Big)^2 \Big] = \frac{E[(x-\mu)^2]}{\sigma^4} = \frac{1}{\sigma^2}
\end{aligned}
$$

### Example 

Show that $\bar{X}$ is an efficient estimator of $\mu$

$$E[\bar{X}] = \mu \qquad var[\bar{X}] = \frac{\sigma^2}{n}$$

$$var[\hat{\mu}] \geq \frac{1}{nI(\mu)} = \frac{\sigma^2}{n}$$

Therefore, $\bar{X}$ is an efficient estimator of $\mu$

### Example 

Show that $s^2$ is an efficient estimator of $\sigma^2$

$$E[s^2] = \sigma^2 \qquad var[s^2] = \frac{2\sigma^4}{n-1}$$

$$\frac{\partial \log f(x)}{\partial \sigma^2} = -\frac{1}{2\sigma^2} + \frac{1}{2\sigma^4}(x-\mu)^2 \qquad \frac{\partial^2 \log f(x)}{\partial \sigma^4} = \frac{1}{2\sigma^4} - \frac{1}{(\sigma^2)^3}(x-\mu)^2$$

$$I(\sigma^2) = -E\Big[ \frac{\partial^2 \log f(x)}{\partial \sigma^4} \Big] = -E\Big[ \frac{1}{2\sigma^4} - \frac{1}{(\sigma^2)^3}(x-\mu)^2 \Big] = -\frac{1}{2\sigma^4} + \frac{1}{\sigma^4} = \frac{1}{2\sigma^4}$$

$$var[\hat{\sigma^2}] = \geq \frac{1}{nI(\sigma^2)} = \frac{1}{n\frac{1}{2\sigma^4}} = \frac{2\sigma^4}{n}$$

Even if $s^2$ does not achieve Cramér-Rao lower bound, it is still called an efficient estimator because it has the minimum variance among all unbiased estimators.

## Relatively efficiency

### Definition 

If $\hat{\sigma_1}$ and $\hat{\sigma_2}$ are both unbiased estimators of a parameter $\theta$. We say that $\hat{\sigma_1}$ is relatively more efficient if $var[\hat{\sigma_1}] < var[\hat{\sigma_2}]$. The ratio $\frac{var[\hat{\sigma_1}]}{var[\hat{\sigma_2}]}$ is a measure of the relative efficiency of $\hat{\sigma_2}$ w.r.t. $\hat{\sigma_1}$.

### Example

Suppose $X_1, X_2, \dots, X_n$ is an i.i.d. random sample from a Poisson distribution with parameter $\lambda$. Let $\hat{\lambda}_1 = \bar{X}$ and $\hat{\lambda}_2 = \frac{X_1 + X_2}{2}$ be two unbiased estimators of $\lambda$. Find the relative efficiency of $\hat{\lambda}_2$ w.r.t. $\hat{\lambda}_1$. 

$$
\begin{aligned}
&var[\hat{\lambda}_1] = var[\bar{X}] = \frac{\sigma^2}{n} = \frac{\lambda}{n} \\
&var[\hat{\lambda}_2] = \frac{1}{4}(var[X_1] + var[X_2]) = \frac{\lambda}{2} \\
&\frac{var[\hat{\lambda}_1]}{var[\hat{\lambda}_2]} = \frac{\lambda/n}{\lambda/2} = \frac{2}{n}
\end{aligned}
$$

## Consistent estimators

### Definition

The estimator $\hat{\theta}$ of a parameter $\theta$ is said to be a consistent estimator if for any $\epsilon$

$$\lim_{n\rightarrow \infty} P(|\hat{\theta} - \theta| \leq \epsilon) = 1 \qquad \text{OR} \qquad \lim_{n\rightarrow \infty} P(|\hat{\theta} - \theta| > \epsilon) = 0$$

We say that $\hat{\theta}$ converges in probability to $\theta$ (also known as the weak law of large numbers). In other words, the average of many independent random variables should be very close to the true mean $\mu$ with high probability.

### Theorem

An unbiased estimator $\hat{\theta}$ of a parameter $\theta$ is consistent if $var[\hat{\theta}] = 0$ as $n \rightarrow \infty$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

Use Chebyshev's Inequality: If a random variable $X$ has mean $\mu$ and variance $\sigma^2$, then for any $k > 1$:

$$P(|X-\mu| \geq k\sigma) \leq \frac{1}{k^2} \qquad \text{OR} \qquad P(|X-\mu| < k\sigma) \geq 1 - \frac{1}{k^2}$$

Let $\epsilon = k\sqrt{var[\hat{\theta}]}$, so $k = \frac{\epsilon}{\sqrt{var[\hat{\theta}]}}$, then

$$P(|\hat{\theta} - \theta| > k\sqrt{var[\hat{\theta}]}) < \frac{1}{k^2} = \frac{var[\hat{\theta}]}{\epsilon^2}$$

$var[\hat{\theta}] \rightarrow 0$ as $n \rightarrow \infty$, then $\frac{var[\hat{\theta}]}{\epsilon^2} \rightarrow 0$. Thus $\hat{\theta}$ is a consistent estimator of $\theta$

</div>

### Example 

Let $X_1, X_2, \dots, X_n$ be i.i.d. random variables with mean $\mu$ and variance $\sigma^2$, is $\bar{X}$ a consistent estimator of $\mu$?

$var[\bar{X}] = \frac{\sigma^2}{n} \rightarrow 0$ as $n \rightarrow \infty$, so $\bar{X}$ is a consistent estimator of $\mu$.

### Example 

Let $X_1, X_2, \dots, X_n$ be random variables with mean $\mu$, variance $\sigma^2$, and $cov[X_i, X_j] = \rho\sigma^2$, is $\bar{X}$ a consistent estimator of $\mu$?

$$var[\bar{X}] = var\Big[ \frac{\sum_{i=1}^n X_i}{n} \Big] = var\Big[ \frac{\mathbf{1}^T\mathbf{X}}{n} \Big] = \frac{1}{n^2} \mathbf{1}^T \boldsymbol{\Sigma} \mathbf{1} =  \frac{\sigma^2}{n^2} \mathbf{1}^T ((1-\rho)\mathbf{I} + \rho \mathbf{1} \mathbf{1}^T) \mathbf{1} = \frac{\sigma^2}{n}((n-1)\rho + 1)$$

$var[\bar{X}] \rightarrow \rho \sigma^2$ as $n \rightarrow \infty$, so $\bar{X}$ is a consistent estimator of $\mu$ iff $\rho = 0$.

## MSE and bias:

Bias of an estimator $\hat{\theta}$ is given by $bias = E[\hat{\theta}] - \theta$

In general, given two unbiased estimators, we would choose the estimator with the smaller variance. However, this is not always possible (there may exist biased estimators with smaller variance). We use the mean square error (MSE):

$$MSE = E[(\hat{\theta} - \theta)^2]$$

as a measure of the goodness of an estimator. MSE can be decomposed as the following form

$$
\begin{aligned}
MSE &= E[(\hat{\theta} - \theta)^2] \\
&= E[(\hat{\theta} - E[\hat{\theta}] + E[\hat{\theta}] - \theta)^2] \\
&= E[(\hat{\theta} - E[\hat{\theta}])^2] + (E[\hat{\theta}] - \theta)^2 + 2(E[\hat{\theta}] - \theta) \overbrace{E[(\hat{\theta} - E[\hat{\theta}])]}^{\text{equal to } 0} \\
&= var[\hat{\theta}] + bias^2
\end{aligned}
$$

### Example - from *Mathematical Statistics with Application* by Wackerly, Mendenhall, Scheaffer

The reading on a voltage meter connected to a test circuit is uniformly distributed over the interval $(\theta, \theta + 1)$, where $\theta$ is the true but unknown voltage of the circuit. Suppose $X_1, X_2, \dots, X_n$ denotes a random sample of such reading:

#### (a) Show that $\bar{X}$ is a biased estimator of $\theta$, and compute the bias.

$$E[\bar{X}] = \mu = \frac{\theta + \theta + 1}{2} = \theta + \frac{1}{2} \qquad bias = \frac{1}{2}$$

#### (b) Find a function of $\bar{X}$ that is an unbiased estimator of $\theta$.

$$\hat{\theta} = \bar{X} - \frac{1}{2} \qquad E[\hat{\theta}] = \theta + \frac{1}{2} - \frac{1}{2} = \theta$$

#### (c) Find the MSE when $\bar{X}$ is used as an estimator of $\theta$.

$$MSE(\bar{X}) = var[\bar{X}] + bias^2 = \frac{\sigma^2}{n} + (\frac{1}{2})^2 = \frac{1}{12n} + \frac{1}{4}$$

#### (d) Find the MSE when the bias corected estimator is used.

$$MSE(\hat{\theta}) = var[\hat{\theta}] = var[\bar{X} - \frac{1}{2}] = var[\bar{X}] = \frac{\sigma^2}{n} = \frac{1}{12n}$$

### Example - from *Theoretical Statistics* by Robert W. Keener

Let $X \sim bin(100, p)$. Consider the three estimator, $\hat{p}_1 = \frac{X}{100}, \hat{p}_2 = \frac{X+3}{100}, \hat{p}_3 = \frac{X+3}{106}$. Find the MSE for each estimator

$$
\begin{aligned}
E[\hat{p}_1] &= E\Big[ \frac{X}{100} \Big] = \frac{100p}{100} = p \\
MSE[\hat{p}_1] &= var\Big[ \frac{X}{100} \Big] = \frac{1}{100^2} var[X] = \frac{100p(1-p)}{100^2} = \frac{p(1-p)}{100} \\
E[\hat{p}_2] &= E\Big[ \frac{X+3}{100} \Big] = \frac{100p+3}{100} = p + \frac{3}{100} \\
var[\hat{p}_2] &= var\Big[ \frac{X+3}{100} \Big] = \frac{1}{100^2} var[X+3] = \frac{100p(1-p)}{100^2} = \frac{p(1-p)}{100} \\
MSE[\hat{p}_2] &= \frac{p(1-p)}{100} + \Big( \frac{3}{100} \Big)^2 \\
E[\hat{p}_3] &= E\Big[ \frac{X+3}{106} \Big] = \frac{100p+3}{106} = \frac{50}{53}p + \frac{3}{106} \\
var[\hat{p}_3] &= var\Big[ \frac{X+3}{106} \Big] = \frac{1}{106^2} var[X+3] = \frac{100p(1-p)}{106^2} \\
MSE[\hat{p}_3] &= \frac{100p(1-p)}{106^2} + \Big(\frac{50}{53}p + \frac{3}{106} - p \Big)^2 = \frac{100p(1-p)}{106^2} + \frac{9}{11236}(1-2p)^2
\end{aligned}
$$


---


# Lec 9: Method of maximum likelihood

### Definition (likelihood function)

Suppose $x_1, x_2, \dots, x_n$ is a random sample of size $n$ from a distribution that has parameter $\theta$. The joint probability density (also known as likelihood function) of these $n$ random variables is

$$L(\theta) = f(x_1, x_2, \dots, x_n; \theta) = f(x_1;\theta) \times f(x_2;\theta) \times \dots \times f(x_n;\theta)$$

Since $x_1, x_2, \dots, x_n$ are independent, the likelihood function can be expressed as the product of the marginal densities. In this function, the parameter $\theta$ is unknown and it will be estimated with the method of maximum likelihood. In principle, the method of maximum likelihood consists of selecting the value of $\theta$ that maximizes the likelihood function (the value of $\theta$ that makes the observed data more likely). To maximize the likelihood function w.r.t. $\theta$, it is often easier to maximize the log likelihood function w.r.t. $\theta$. Therefore, we will take the derivative of the log likelihood function w.r.t. $\theta$, set it equal to zero and solve for $\theta$. The result will be denoted with $\hat{\theta}$ and we refer to it as the $MLE$ of the parameter $\theta$.

### Example

#### (a) Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} \text{Poisson}(\lambda)$. Find the MLE of $\lambda$

$$
\begin{aligned}
L(\lambda) &= p(x_1, x_2, \dots, x_n) = \prod_{i=1}^n p(x_i) = \prod_{i=1}^n \frac{\lambda^{x_i}e^{-\lambda}}{x_i!} &= \frac{\lambda^{\sum_{i=1}^n x_i}e^{-n\lambda}}{\prod_{i=1}^n x_i!} \\
\log L(\lambda) &= (\sum_{i=1}^n x_i) \log \lambda - n\lambda - \log \prod_{i=1}^n x_i! \\
\frac{d \log L(\lambda)}{d\lambda} &= \frac{\sum_{i=1}^n x_i}{\lambda} - n = 0
\quad \Rightarrow \quad \hat{\lambda} = \frac{\sum_{i=1}^n X_i}{n} = \bar{X}
\end{aligned}
$$

#### (b) Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} exp(\lambda)$. Find the MLE of $\lambda$

$$
\begin{aligned}
L(\lambda) &= p(x_1, x_2, \dots, x_n) = \prod_{i=1}^n p(x_i) = \prod_{i=1}^n \lambda e^{-\lambda x_i} = \lambda^n e^{-\lambda \sum_{i=1}^n x_i} \\
\log L(\lambda) &= n\log \lambda - \lambda \sum_{i=1}^n x_i \\
\frac{d \log L(\lambda)}{d\lambda} &= \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0
\quad \Rightarrow \quad \hat{\lambda} = \frac{n}{\sum_{i=1}^n X_i} = \frac{1}{\bar{X}}
\end{aligned}
$$

Note: $\hat{\lambda} = \frac{1}{\bar{X}}$ is biased

exponential distribution is a special case of gamma distribution: $\sum_{i=1}^n X_i \sim \Gamma(n, \frac{1}{\lambda})$ 

$$E[\hat{\lambda}] = E\Big[ \frac{n}{\sum_{i=1}^n x_i} \Big] = nE[ (\sum_{i=1}^n x_i)^{-1} ] = n\frac{\Gamma(n-1)(\frac{1}{\lambda})^{-1}}{\Gamma(n)} = n\lambda \frac{\Gamma(n-1)}{(n-1)\Gamma(n-1)} = \frac{n}{n-1}\lambda$$

Thus, $\hat{\theta} = \frac{1}{\bar{X}} \cdot \frac{n-1}{n} = \frac{n-1}{\sum_{i=1}^n X_i}$ is an unbiased estimator of $\lambda$

#### (c) Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} unif(0, \theta)$. Find the MLE of $\theta$

$$
\begin{aligned}
L(\theta) &= f(x_1, x_2, \dots, x_n) = \prod_{i=1}^n f(x_i)
= \prod_{i=1}^n \frac{1}{\theta - 0} = \frac{1}{\theta^n} \\
\log L(\theta) &= -n\log \theta \\
\frac{d \log L(\theta)}{d\theta} &= -\frac{n}{\theta} = 0
\end{aligned}
$$

We say the MLE of $\theta$ is $\hat{\theta} = \max(X_1, X_2, \dots, X_n)$ or $\hat{\theta} = X_{(n)}$.

Find the pdf of $X_{(n)}$

$$
\begin{aligned}
F_{X_{(n)}} &= P(X_{(n)} \leq x) = P(X_1 \leq x; \dots; X_n \leq x) \overset{\text{indep}}{=} P(X_1 \leq x) \dots P(X_n \leq x) = (F(x))^n \\
g_{X_{(n)}} &= n(F(x))^{n-1}f(x) = n(\frac{x}{\theta - 0})^{n-1}\frac{1}{\theta - 0} = \frac{n}{\theta^n} x^{n-1} \\
E[\hat{\theta}] &= E[X_{(n)}] = \int_0^\theta x\frac{n}{\theta^n} x^{n-1} = \frac{n}{n+1}\theta
\end{aligned}
$$

Thus, $\frac{n+1}{n}X_{(n)}$ is an unbiased estimator.

#### (d) Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$. Find the MLE of $\mu$ and $\sigma^2$

$$
\begin{aligned}
&L(\mu, \sigma^2) = f(x_1, x_2, \dots, x_n) = \prod_{i=1}^n f(x_i) 
= \prod_{i=1}^n (2\pi\sigma^2)^{-\frac{1}{2}}e^{-\frac{(x_i-\mu)^2}{2\sigma^2}} 
= (2\pi\sigma^2)^{-\frac{n}{2}}e^{-\frac{\sum_{i=1}^n (x_i-\mu)^2}{2\sigma^2}} \\
&\log L(\mu, \sigma^2) = -\frac{n}{2} \log(2\pi\sigma^2) - \frac{\sum_{i=1}^n (x_i-\mu)^2}{2\sigma^2} \\
&\frac{\partial \log L(\mu)}{\partial \mu} = \frac{2}{2\sigma^2} \sum_{i=1}^n (x_i-\mu) = 0
\quad \Rightarrow \quad \sum_{i=1}^n x_i - n\mu = 0
\quad \Rightarrow \quad \hat{\mu} = \bar{X} \\
&\frac{\partial \log L(\sigma^2)}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4} \sum_{i=1}^n (x_i-\mu)^2 = 0 
\quad \Rightarrow \quad \hat{\sigma^2} = \frac{\sum_{i=1}^n (x_i-\hat{\mu})^2}{n} = \frac{\sum_{i=1}^n (x_i-\bar{X})^2}{n} \\
E[\hat{\sigma^2}] &= \frac{1}{n}E[\sum_{i=1}^n (x_i-\bar{X})^2] = \frac{(n-1)\sigma^2}{n} \qquad \text{(biased)} \\
s^2 &= \frac{n}{n-1}\hat{\sigma^2} \qquad \text{(unbiased)}
\end{aligned}
$$

#### (e) Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu_1, \sigma^2)$ and $Y_1, Y_2, \dots, Y_m \overset{\text{i.i.d.}}{\sim} N(\mu_2, \sigma^2)$. Find the MLE of $\mu$ and $\sigma^2$

$$
\begin{aligned}
L(\mu_1, \mu_2, \sigma^2) &= f(x_1, x_2, \dots, x_n) \cdot g(y_1, y_2, \dots, y_m) \\
&= \prod_{i=1}^n f(x_i) \prod_{j=1}^m g(y_j) \\
&= \prod_{i=1}^n (2\pi\sigma^2)^{-\frac{1}{2}}e^{-\frac{(x_i-\mu_1)^2}{2\sigma^2}} 
\prod_{j=1}^m (2\pi\sigma^2)^{-\frac{1}{2}}e^{-\frac{(y_j-\mu_2)^2}{2\sigma^2}} \\
&= (2\pi\sigma^2)^{-\frac{n}{2}}e^{-\frac{\sum_{i=1}^n (x_i-\mu_1)^2}{2\sigma^2}} 
(2\pi\sigma^2)^{-\frac{m}{2}}e^{-\frac{\sum_{j=1}^m (y_j-\mu_2)^2}{2\sigma^2}} \\
\log L(\mu_1, \mu_2, \sigma^2) &= 
-\frac{n}{2}\log(2\pi\sigma^2) -\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu_1)^2
-\frac{m}{2}\log(2\pi\sigma^2) -\frac{1}{2\sigma^2}\sum_{j=1}^m (y_j-\mu_2)^2 \\
\frac{\partial \log L(\mu_1)}{\partial \mu_1} &= \frac{2}{2\sigma^2}\sum_{i=1}^n (x_i-\mu_1) = 0 
\quad \Rightarrow \quad \hat{\mu_1} = \bar{X} \\
\frac{\partial \log L(\mu_2)}{\partial \mu_2} &= \frac{2}{2\sigma^2}\sum_{j=1}^m (y_j-\mu_2) = 0 
\quad \Rightarrow \quad \hat{\mu_2} = \bar{Y} \\
\frac{\partial \log L(\sigma^2)}{\partial \sigma^2} &= 
-\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4} \sum_{i=1}^n (x_i-\mu_1)^2 
-\frac{m}{2\sigma^2} + \frac{1}{2\sigma^4} \sum_{j=1}^m (y_j-\mu_2)^2 = 0 \\
\Rightarrow \quad \hat{\sigma^2} 
&= \frac{\sum_{i=1}^n (X_i-\hat{\mu_1})^2 + \sum_{j=1}^m (Y_j-\hat{\mu_2})^2}{n + m} 
= \frac{\sum_{i=1}^n (X_i-\bar{X})^2 + \sum_{j=1}^m (Y_j-\bar{Y})^2}{n + m}
= \frac{(n-1)s^2_X + (m-1)s^2_Y}{n + m} \\
E[\hat{\sigma^2}] &= \frac{(n-1)\sigma^2 + (m-1)\sigma^2}{n + m} = \frac{(n+m-2)\sigma^2}{n+m} \qquad \text{(biased)} \\
s^2_{pool} &= \frac{n+m}{n+m-2}\hat{\sigma^2} \qquad \text{(unbiased)}
\end{aligned}
$$

two sample difference t-test:

$$\bar{X} - \bar{Y} \sim N(\mu_1 - \mu_2, \sigma^2(\frac{1}{n} + \frac{1}{m}))$$

$$\frac{Z}{\sqrt{U/df}} = \frac{ \frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\sigma\sqrt{\frac{1}{n} + \frac{1}{m}}} }{ \sqrt{\frac{(n+m-2)s^2_{pool}}{\sigma^2} / (n+m-2)} } = \frac{(\bar{X} - \bar{Y}) - (\mu_1 - \mu_2)}{\sqrt{s^2_{pool}(\frac{1}{n} + \frac{1}{m})}} \sim t_{n+m-2}$$

### Theorem

**Asymptotic efficiency of maximum likelihood estimates**: Let $X_1, X_2, \dots, X_n$ be i.i.d. random variables from a probability density function $f(x \mid \theta)$. Then if $\hat{\theta}$ is the MLE of $\theta$, then $\hat{\theta} \sim N(\theta, \frac{1}{nI(\theta)})$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

Start with the likelihood function

$$
\begin{aligned}
L(\theta) &= \prod_{i=1}^n f(x_i \mid \theta) \\
\log L(\theta) &= \sum_{i=1}^n \log f(x_i \mid \theta) \\
\frac{\partial}{\partial \theta} \log L(\theta) &= \sum_{i=1}^n \frac{\partial}{\partial \theta} \log f(x_i \mid \theta) 
\end{aligned}
$$

Let $\hat{\theta}$ be the MLE of $\theta$, the first order Taylor expansion at $\hat{\theta}$ is:

$$
\begin{aligned}
\sum_{i=1}^n \frac{\partial}{\partial \theta} \log f(x_i \mid \theta) &\approx \sum_{i=1}^n \frac{\partial}{\partial \theta} \log f(x_i \mid \hat{\theta}) + \Big[ \sum_{i=1}^n \frac{\partial^2}{\partial \theta^2} \log f(x_i \mid \hat{\theta}) \Big] (\theta - \hat{\theta}) \\
\frac{1}{\sqrt{n}} \sum_{i=1}^n \frac{\partial}{\partial \theta} \log f(x_i \mid \theta) &\approx \frac{1}{\sqrt{n}} \sum_{i=1}^n \frac{\partial}{\partial \theta} \log f(x_i \mid \hat{\theta}) + \Big[ \frac{1}{\sqrt{n}} \sum_{i=1}^n \frac{\partial^2}{\partial \theta^2} \log f(x_i \mid \hat{\theta}) \Big] (\theta - \hat{\theta}) 
\end{aligned}
$$

The first term on the right hand side is zero (because this is what we do to find $\hat{\theta}$). Therefore, the relationship is reduced to the following:

$$
\frac{1}{\sqrt{n}} \sum_{i=1}^n \frac{\partial}{\partial \theta} \log f(x_i \mid \theta) \approx  \Big[ \frac{1}{\sqrt{n}} \sum_{i=1}^n \frac{\partial^2}{\partial \theta^2} \log f(x_i \mid \hat{\theta}) \Big] (\theta - \hat{\theta})
$$

Examine the left hand side: This involves the sum of $n$ i.i.d. forms (central limit theorem). Each one of these has mean zero and variance $I(\theta)$. Therefore, the left hand side follows approximately $N(0, I(\theta))$. Therefore

$$
\begin{aligned}
&\Big[ \frac{1}{\sqrt{n}} \sum_{i=1}^n \frac{\partial^2}{\partial \theta^2} \log f(x_i \mid \hat{\theta}) \Big] (\theta - \hat{\theta}) \sim N(0, I(\theta)) \\
&\Big[ -\frac{1}{n} \sum_{i=1}^n \frac{\partial^2}{\partial \theta^2} \log f(x_i \mid \hat{\theta}) \Big] \sqrt{n}(\theta - \hat{\theta}) \sim N(0, I(\theta)) 
\end{aligned}
$$

The expression in the bracket converges to $I(\theta)$ (law of large numbers) and therefore

$$I(\theta)\sqrt{n}(\theta - \hat{\theta}) \sim N(0, I(\theta))$$

$$\hat{\theta} \sim N(\theta, \frac{1}{nI(\theta)})$$

</div>

## Application of MLE in linear model

### Intercept model

$$Y_i = \mu + \epsilon_i$$

with $\epsilon_1, \epsilon_2, \dots, \epsilon_n \overset{\text{i.i.d.}}{\sim} N(0, \sigma^2)$

$$
\begin{aligned}
&\min S = \min \sum_{i=1}^n \epsilon^2 = \min \sum_{i=1}^n (Y_i - \mu)^2 \\
\Rightarrow \quad &\frac{dS}{d\mu} = -2\sum_{i=1}^n (Y_i - \mu) = 0 \\
\Rightarrow \quad &\sum_{i=1}^n Y_i - n\mu = 0 \\
\Rightarrow \quad &\hat{\mu} = \bar{Y}
\end{aligned}
$$

If $Y_1, Y_2, \dots, Y_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$, then MLE of $\mu$ is $\hat{\mu} = \bar{Y}$

### Simple linear model

$$Y_i = \beta_0 + \beta_1 X_i + \epsilon_i$$

with $\epsilon_1, \epsilon_2, \dots, \epsilon_n \overset{\text{i.i.d.}}{\sim} N(0, \sigma^2)$. $Y_i$ are not identically distributed because mean changes. $f(y_i) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(y_i - \beta_0 - \beta_1 x_i)^2}{2\sigma^2}}$.

$$
\begin{aligned}
L(\sigma^2, \beta_1, \beta_2) &= \prod_{i=1}^n f(y_i) = (2\pi\sigma^2)^{-\frac{n}{2}} e^{-\frac{\sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2}{2\sigma^2}} \\
\log L(\sigma^2, \beta_1, \beta_2) &= -\frac{n}{2} \log(2\pi\sigma^2) -\frac{\sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2}{2\sigma^2} \\
\frac{\partial L(\sigma^2, \beta_1, \beta_2)}{\partial \beta_0} &= \frac{2}{2\sigma^2} \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i) = 0 \quad \Rightarrow \quad n\beta_0 + \beta_1 \sum_{i=1}^n x_i = \sum_{i=1}^n y_i \\
\frac{\partial L(\sigma^2, \beta_1, \beta_2)}{\partial \beta_1} &= \frac{2}{2\sigma^2} \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)x_i = 0 \quad \Rightarrow \quad \beta_0 \sum_{i=1}^n x_i + \beta_1 \sum_{i=1}^n x_i^2 = \sum_{i=1}^n x_i y_i \\
\hat{\beta_0} &= \bar{Y} - \hat{\beta_1} \bar{X} \\
\hat{\beta_1} &= \frac{\sum_{i=1}^n X_i Y_i -\frac{1}{n}(\sum_{i=1}^n X_i)(\sum_{i=1}^n Y_i)}{\sum_{i=1}^n X_i^2 - \frac{(\sum_{i=1}^n X_i)^2}{n}} \\
&= \frac{\sum_{i=1}^n (X_i - \bar{X}) \sum_{i=1}^n (Y_i - \bar{Y})}{\sum_{i=1}^n (X_i - \bar{X})^2} \\
&= \frac{\sum_{i=1}^n (X_i - \bar{X}) Y_i}{\sum_{i=1}^n (X_i - \bar{X})^2} \qquad \text{since } \bar{Y}\sum_{i=1}^n (X_i - \bar{X}) \\
&= \frac{(X_1 - \bar{X})}{\sum_{i=1}^n (X_i - \bar{X})^2}Y_1 + \dots + \frac{(X_n - \bar{X})}{\sum_{i=1}^n (X_i - \bar{X})^2}Y_n \\
&\text{Any linear combination of jointly normal random variables is normally distributed} \\
\frac{\partial L(\sigma^2, \beta_1, \beta_2)}{\partial \sigma^2} &= -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4} \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2 = 0 \\
\hat{\sigma^2} &= \frac{1}{n} \sum_{i=1}^n (y_i - \hat{\beta_0} - \hat{\beta_1} x_i)^2 = \frac{1}{n} \sum_{i=1}^n e_i 
\end{aligned}
$$

**Line of fitted value**: $\hat{Y_i} = \hat{\beta_0} + \hat{\beta_1} X_i = \bar{Y} - \hat{\beta_1} \bar{X} + \hat{\beta_1} = \bar{Y} + \hat{\beta_1}(X_i - \bar{X})$

**Residuals**: $e_i = Y_i - \hat{Y_i} = Y_i - \bar{Y} + \hat{\beta_1}(X_i - \bar{X}) \quad \Rightarrow \quad \sum e_i = 0; \quad \sum e_iX_i = 0; \quad \sum e_iY_i = 0$

$$
\begin{aligned}
E[\hat{\beta_1}] &= E\Big[ \frac{\sum_{i=1}^n (X_i - \bar{X}) Y_i}{\sum_{i=1}^n (X_i - \bar{X})^2} \Big] \\
&= \frac{\sum_{i=1}^n (X_i - \bar{X}) E[Y_i]}{\sum_{i=1}^n (X_i - \bar{X})^2} \\
&= \frac{\sum_{i=1}^n (X_i - \bar{X}) (\beta_0 + \beta_1 X_i)}{\sum_{i=1}^n (X_i - \bar{X})^2} \\
&= \frac{\beta_0 \overbrace{\sum_{i=1}^n (X_i - \bar{X})}^{\text{equal to } 0} + \beta_1 \overbrace{\sum_{i=1}^n (X_i - \bar{X})\bar{X}}^{=\sum_{i=1}^n (X_i - \bar{X})^2}}{\sum_{i=1}^n (X_i - \bar{X})^2} = \beta_1 \qquad \text{(unbiased)} \\
var[\hat{\beta_1}] &= var\Big[ \frac{\sum_{i=1}^n (X_i - \bar{X}) Y_i}{\sum_{i=1}^n (X_i - \bar{X})^2} \Big] 
= \frac{\sum_{i=1}^n (X_i - \bar{X})^2 var[Y_i]}{(\sum_{i=1}^n (X_i - \bar{X})^2)^2} 
= \frac{\sigma^2}{\sum_{i=1}^n (X_i - \bar{X})^2} \\
E[\hat{\beta_0}] &= E[\bar{Y} - \hat{\beta_1} \bar{X}] 
= E[\bar{Y}] - \bar{X} E[\hat{\beta_1}] 
= \beta_0 + \beta_1 \bar{X} - \beta_1 \bar{X}
= \beta_0 \qquad \text{(unbiased)} \\
var[\hat{\beta_0}] &= var[\bar{Y} - \hat{\beta_1} \bar{X}] 
= var[\bar{Y}] + \bar{X}^2 var[\hat{\beta_1}] - 2\bar{X} cov[\bar{Y}, \hat{\beta_1}]
= \sigma^2(\frac{1}{n} + \frac{\bar{X}^2}{\sum_{i=1}^n (X_i - \bar{X})^2}) \\
\text{Note: } cov[\bar{Y}, \hat{\beta_1}] &= cov\Big[ \frac{\sum_{i=1}^n Y_i}{n}, \frac{\sum_{i=1}^n (X_i - \bar{X}) Y_i}{\sum_{i=1}^n (X_i - \bar{X})^2} \Big] \\
&= cov\Big[ \frac{\sum_{i=1}^n Y_i}{n}, \frac{(X_1 - \bar{X})}{\sum_{i=1}^n (X_i - \bar{X})^2}Y_1 + \dots + \frac{(X_n - \bar{X})}{\sum_{i=1}^n (X_i - \bar{X})^2}Y_n \Big] \qquad \text{use bilinearity property} \\
&= \frac{\sum_{i=1}^n Y_i}{n}, \frac{\sigma^2(X_1 - \bar{X})}{\sum_{i=1}^n (X_i - \bar{X})^2} + \dots + \frac{\sigma^2(X_n - \bar{X})}{\sum_{i=1}^n (X_i - \bar{X})^2} \\
&= \frac{\sigma^2\sum_{i=1}^n (X_i - \bar{X})}{\sum_{i=1}^n (X_i - \bar{X})^2} = 0
\end{aligned}
$$

Therefore,

$$E[\hat{\sigma^2}] = E\Big[ \frac{\sum_{i=1}^n e_i^2}{n} \Big] = \frac{n-2}{n}\sigma^2 \qquad \text{(biased)}$$
$$s^2 = \frac{\sum_{i=1}^n e_i^2}{n-2}; \qquad E[s^2] = \sigma^2 \qquad \text{(unbiased)}$$

$$\hat{\beta_1} \sim N(\beta_1, \frac{\sigma^2}{\sum_{i=1}^n (X_i - \bar{X})^2}); \qquad 
\frac{(n-2)s^2}{\sigma^2} \sim \chi^2_{n-2}; \qquad \beta_1 \perp\!\!\!\perp s^2$$


---


# Lec 10: Order statistics

Let $X_1, X_2, \dots, X_n$ denote independent continuous random variables with cdf $F(x)$ and pdf $f(x)$. Denote the ordered random variables with $X_{(1)}, X_{(2)}, \dots, X_{(n)}$ where $X_{(1)} \leq X_{(2)} \leq \dots \leq X_{(n)}$ or $X_{(1)} = \min(X_1, X_2, \dots, X_n)$ and $X_{(n)} = \max(X_1, X_2, \dots, X_n)$. Similarly, $X_{(j)}$ is the $j_{th}$ ordedr statistic. 

## Probability density function of the $j_{th}$ order statistic

$$g_{X_{(j)}}(x) = \frac{n!}{(n-j)! (j-1)!} [F_X(x)]^{j-1} [1 - F_X(x)]^{n-1} f_X(x)$$

- Probability density function of the $1_{st}$ order statistic: $g_{X_{(1)}}(x) = n[1 - F_X(x)]^{n-1} f_X(x)$
- Probability density function of the $n_{th}$ order statistic: $g_{X_{(n)}}(x) = n[F_X(x)]^{n-1} f_X(x)$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">

**Proof**:

We will find the cdf of the $j_{th}$ order statistics and then the pdf by taking the derivative of the cdf. The cdf is denoted by $F_{X_{(j)}}(x) = P(X_{(j)} \leq x)$. Introduce a discrete random variable $Y$ that counts the number of variables less than or equal to $x$. The statement $P(X_{(j)} \leq x) = P(Y \geq j)$. If we call "success" the event $X_i < x$, then $Y \sim bin(n,p) = bin(n,F_X(x))$

$$
\begin{aligned}
F_{X_{(j)}}(x) &= P(X_{(j)} \leq x) = P(Y \geq j) \\
&= \sum_{k=j}^n \binom{n}{k} p^k (1-p)^{n-k} \\
&= \sum_{k=j}^n \binom{n}{k} F_X(x)^k (1-F_X(x))^{n-k} \\
g_{X_{(j)}}(x) &= \frac{dF_{X_{(j)}}(x)}{dx} \\
&= \sum_{k=j}^n \binom{n}{k} kF_X(x)^{k-1} f_X(x) (1-F_X(x))^{n-k} \\
&- \sum_{k=j}^n \binom{n}{k} (n-k)F_X(x)^k (1-F_X(x))^{n-k-1} f_X(x) \\
&= \binom{n}{j} jf_X(x) F_X(x)^{j-1} (1-F_X(x))^{n-j} \qquad \text{(when } k=j \text{ )} \\
&+ \sum_{k=j+1}^n \binom{n}{k} kF_X(x)^{k-1} f_X(x) (1-F_X(x))^{n-k} \\ 
&- \sum_{k=j}^{n-1} \binom{n}{k} (n-k)F_X(x)^k (1-F_X(x))^{n-k-1} f_X(x) \qquad \text{(last term is zero when when } k=n \text{ )} \\
&= \frac{n!}{(n-j)!j!}jf_X(x) F_X(x)^{j-1} (1-F_X(x))^{n-j} \\
&+ \sum_{k=j}^{n-1} \binom{n}{k+1} (k+1)F_X(x)^{k} f_X(x) (1-F_X(x))^{n-k-1} \\ 
&- \sum_{k=j}^{n-1} \binom{n}{k} (n-k)F_X(x)^k (1-F_X(x))^{n-k-1} f_X(x) \\
&= \frac{n!}{(n-j)!(j-1)!} F_X(x)^{j-1} (1-F_X(x))^{n-j} f_X(x)
\end{aligned}
$$

Note: $\binom{n}{k+1}(K+1) = \binom{n}{k}(n-k)$, so the last 2 terms before the last line cancel.

</div>

## Joint probability density function of $X_{(1)}, X_{(2)}, \dots, X_{(n)}$

$$g_{X_{(1)}, X_{(2)}, \dots, X_{(n)}}(x_1, x_2, \dots, x_n) = n!f_X(x_1)f_X(x_2)\dots f_X(x_n)$$

## Joint probability density function of $X_{(i)}$ and $X_{(j)}$, with $1 \leq i < j \leq n$

$$g_{X_{(i)}, X_{(j)}}(u,v) = \frac{n!}{(i-1)!(j-1-i)!(n-j)!} f_X(u) f_X(v) [F_X(u)]^{i-1} [F_X(v) - F_X(u)]^{j-1-i} [1 - F_X(v)]^{n-j}$$

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">

**Proof**:

An intuitive derivation of the density function of the $j_{th}$ order statistics. This intuitive derivation is based on this result $P(y \leq Y \leq y + dy) \approx f(y) dy$. Consider the $j_{th}$ order statistic $X_{(j)}$. if $X_{(j)}$ is in the neighborhood of $x$, then there are
- $j-1$ random variables less than $x$, each one with probability $p_1 = P(X \leq x) = F_X(x)$,
- $1$ random variable near $x$, with probability $p_2 = P(x \leq X \leq x + dx) \approx f_X(x) dx$, and
- $n-j$ random variables larger than $x$, with probability $p_3 = P(X > x) = 1 - P(X \leq x) = 1 - F_X(x)$.

Therefore,

$$
\begin{aligned}
P(x \leq X_{(j)} \leq x + dx) &\approx g_{X_{(j)}}(x)dx \\
&= \binom{n!}{j-1, 1, n-j} p_1^{j-1} p_2 p_3^{n-j} \qquad \text{multinomial distribution} \\
&= \frac{n!}{(j-1)! (n-j)!} F_X(x)^{j-1} f_X(x) dx [1-F_X(x)]^{n-j} \\
\Rightarrow \quad g_{X_{(j)}}(x) &= \frac{n!}{(j-1)! (n-j)!} F_X(x)^{j-1} f_X(x) [1-F_X(x)]^{n-j}
\end{aligned}
$$

Using this intuitive derivation, we can now find the joint probability density function of $X_{(i)}$ and $X_{(j)}$. Using the same approximation as above. For $u < v$, we need to have the following arrangement:
- $i-1$ random variables less than $u$, each one with probability $p_1 = P(X \leq u) = F_X(u)$
- $1$ random variable near $u$ with probability $p_2 = P(u \leq X \leq u + du) \approx f_X(u) du$
- $j-1-i$ random variables between $u$ and $v$ with probability $p_3 = P(u \leq X \leq v) = F_X(v) - F_X(u)$
- $1$ random variable near $v$ with probability $p_4 = P(v \leq X \leq v + dv) \approx f_X(v) dv$
- $n-j$ random variables larger than $v$, each one with probability $p_5 = P(X > v) = 1 - F_X(v)$

$$
\begin{aligned}
&\quad P(u \leq X \leq u + du, v \leq X \leq v + dv) \approx g_{X_{(i)}, X_{(j)}}(u,v) du dv \\
&= \binom{n!}{i-1, 1, j-1-i, 1, n-j} p_1^{i-1} p_2 p_3^{j-1-i} p_4 p_5^{n-j} \\
&= \frac{n!}{(i-1)!(j-1-i)!(n-j)!} F_X(u)^{i-1} f_X(u) du [F_X(v) - F_X(u)]^{j-1-i} f_X(v) dv [1 - F_X(v)]^{n-j} \\
g_{X_{(i)}, X_{(j)}}(u,v) &= \frac{n!}{(i-1)!(j-1-i)!(n-j)!} F_X(u)^{i-1} f_X(u) [F_X(v) - F_X(u)]^{j-1-i} f_X(v) [1 - F_X(v)]^{n-j}
\end{aligned}
$$

</div>

### Example

Electronic components of a certain type have a length life (in hours) $X$, that follows $exp(\lambda)$. 

#### (a) Suppose that $n$ such components operate independently and in series in a certain system (the system fails when either component fails). Find the density function for the length of life of the system.

$$
\begin{aligned}
g_{X_{(1)}}(x) &= n[1 - F_X(x)]^{n-1} f_X(x) \\
&= n[1 - (1 - e^{-\lambda x})]^{n-1} \lambda e^{-\lambda x} \\
&= n\lambda e^{-n\lambda x} 
\end{aligned}
$$

Therefore, $X_{(1)} \sim exp(n\lambda)$

#### (b) Suppose that $n$ such components operate independently and in parallel in a certain system (the system does not fail until both components fail). Find the density function for the length of life of the system.

$$
\begin{aligned}
g_{X_{(n)}}(x) &= n[F_X(x)]^{n-1} f_X(x) \\
&= n(1 - e^{-\lambda x})^{n-1} \lambda e^{-\lambda x} \\
&= n\lambda e^{-\lambda x} (1 - e^{-\lambda x})^{n-1}
\end{aligned}
$$

### Example

Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} unif(0, \theta)$. Find the pdf of $X_{(1)}$, $X_{(i)}$, $X_{(j)}$, and the joint pdf of $X_{(1)}$, $X_{(n)}$.

$$
\begin{aligned}
g_{X_{(1)}}(x) &= n\Big(1 - \frac{x}{\theta}\Big)^{n-1} \frac{1}{\theta} \\
g_{X_{(n)}}(x) &= n\Big(\frac{x}{\theta}\Big)^{n-1} \frac{1}{\theta} = \frac{n}{\theta^n} x^{n-1} \\
g_{X_{(j)}}(x) &= \frac{n_!}{(n-j)!(j-1)!} \Big(\frac{x}{\theta}\Big)^{j-1} \Big(1 - \frac{x}{\theta}\Big)^{n-j} \frac{1}{\theta} \\
g_{X_{(1)}, X_{(n)}}(u,v) &= \frac{n!}{(1-1)!(n-1-1)!(n-n)!} \frac{1}{\theta} \frac{1}{\theta} \Big(\frac{u}{\theta}\Big)^{1-1} \Big(\frac{v}{\theta} - \frac{u}{\theta}\Big)^{n-1-1} \Big(1 - \frac{v}{\theta}\Big)^{n-n} \\
&= \frac{n!}{(n-2)!} \frac{1}{\theta^2} \Big(\frac{v}{\theta} - \frac{u}{\theta}\Big)^{n-2} \\
&= \frac{n(n-1)}{\theta^n} (v - u)^{n-2}
\end{aligned}
$$

### Example (revisited)

Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} unif(0, \theta)$. We have seen that the mle of $\theta$ is $\hat{\theta} = X_{(n)}$. Is it biased?

$$
\begin{aligned}
g_{X_{(n)}} &= n[F_X(x)]^{n-1} f_X(x) \\
g_{\hat{\theta}} &= n \Big[ \frac{x - 0}{\theta - 0} \Big]^{n-1} \frac{1}{\theta - 0} = n \frac{x^{n-1}}{\theta^{n-1}} \frac{1}{\theta} = \frac{nx^{n-1}}{\theta^n} \\
E[\hat{\theta}] &= \int_0^\theta x \frac{nx^{n-1}}{\theta^n} dx = \frac{n}{\theta^n} \int_0^\theta x^n dx = \frac{n}{\theta^n} \Big[ \frac{x^{n+1}}{n+1} \Big]_0^\theta = \frac{n}{n+1}\theta \qquad \text{(biased)}
\end{aligned}
$$


---


# Lec 11: Method of moments

The method of moments is based on the assumption that the sample moments are good estimates of the corresponding population moments.

### Definition

| Population moments | Sample moments |
|--------------------|----------------|
| $E[X] = \mu$ is the first population moment | $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is the first sample moment |
| $E[X^2]$ is the second population moment | $\frac{1}{n}\sum_{i=1}^n X_i^2$ is the second sample moment |
| $\vdots$ | $\vdots$ |
| $E[X^k]$ is the $k_{th}$ population moment | $\frac{1}{n}\sum_{i=1}^n X_i^k$ is the $k_{th}$ sample moment |

Therefore, $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is a good estimator of $E[X] = \mu$. Similarly, $\frac{1}{n}\sum_{i=1}^n X_i^2$ is a good estimator of $E[X^2]$, etc.

### Example

#### (a) Let $X_1, X_2, \dots, X_n$ denote an i.i.d. random sample from a Poisson distribution with mean $\lambda$. Find the moment estimator of $\lambda$.

$$E[X] = \lambda \quad \Rightarrow \quad \hat{\lambda} = \bar{X}$$

#### (b) Let $X_1, X_2, \dots, X_n$ denote an i.i.d. random sample from unif(0, \theta). Find the method of moments estimator of $\theta$.

$$
\begin{aligned}
E[X] &= \frac{\theta}{2} \quad \Rightarrow \quad \hat{\theta} = 2\bar{X} \\
E[\hat{\theta}] &= 2E[\bar{X}] = 2\mu = 2 \frac{\theta}{2} = \theta \\
var[\hat{\theta}] &= 4var[\bar{X}] = 4\frac{\sigma^2}{n} = 4\frac{\theta^2}{12n} = \frac{\theta^2}{3n}
\end{aligned}
$$

Note: unbiased estimator: $\hat{\theta} = \frac{n+1}{n}X_{(n)}$

#### (c) Let $X_1, X_2, \dots, X_n$ denote an i.i.d. random sample from $N(\mu, \sigma^2)$. Find the moment estimator of $\mu$ and $\sigma^2$.

$$
\begin{aligned}
E[X] &= \mu \quad \Rightarrow \quad \hat{\mu} = \bar{X} \\
\hat{\sigma^2} &= E[X^2] - E[X]^2 = \frac{1}{n}\sum_{i=1}^n X_i^2 - \bar{X}^2 = \frac{1}{n}(\sum_{i=1}^n X_i^2 - n\bar{X}) = \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2 \\
\end{aligned}
$$

#### (d) Let $X_1, X_2, \dots, X_n$ denote an i.i.d. random sample from $N(0, \sigma^2)$. Find the moment estimator of $\sigma^2$.

$$\hat{\sigma^2} = E[X^2] = \frac{1}{n}\sum_{i=1}^n X_i^2$$

#### (e) Let $X_1, X_2, \dots, X_n$ denote an i.i.d. random sample from the probability density function $f(x;\theta) = (\theta + 1)x^\theta, \quad 0 < x < 1, \quad \theta > -1$. Find the method of moments estimator of $\theta$.

$$
\begin{aligned}
E[X] &= \int_0^1 x(\theta + 1)x^\theta dx = \Big[ (\theta + 1)\frac{x^{\theta+2}}{\theta+2} \Big] = \frac{\theta + 1}{\theta + 2} \\
\Rightarrow \quad \frac{\hat{\theta} + 1}{\hat{\theta} + 2} &= \bar{X} \\
\Rightarrow \quad \hat{\theta} &= \frac{2\bar{X} - 1}{1 - \bar{X}}
\end{aligned}
$$

#### (f) Let $X_1, X_2, \dots, X_n$ denote an i.i.d. random sample from $\Gamma(\alpha, \beta)$. Find the moment estimator of $\alpha$ and $\beta$.

$$
\begin{aligned}
\mu &= E[X] = \alpha\beta \quad \Rightarrow \quad \hat{\alpha}\hat{\beta} = \bar{X} \\
\sigma^2 &= E[X^2] - E[X]^2 = \alpha\beta^2 \quad \Rightarrow \quad \hat{\alpha}\hat{\beta}^2 = \frac{1}{n} \sum_{i=1}^n X_i^2 - (\bar{X})^2 = \frac{1}{n} \sum_{i=1}^2 (X_i - \bar{X})^2 \\
\Rightarrow \quad \hat{\beta} &= \frac{\sum_{i=1}^n (X_i - \bar{X})^2}{n\bar{X}} \\
\Rightarrow \quad \hat{\alpha} &= \frac{\bar{X}}{\hat{\beta}} = \frac{n\bar{X}^2}{\sum_{i=1}^n (X_i - \bar{X})^2}
\end{aligned}
$$


---


# Lec 12: Data reduction and sufficiency principle

### Introduction

We have seen some examples of estimators, some based on intuition or the method of maximum likelihood or the method of moments. For example, $\bar{X}$ is an unbiased estimator of $\mu$, $s^2$ is an unbiased estimator of $\sigma^2$ etc. Are these estimators sufficient, in the sense that we no longer need the actual data values in order for example, to construct a confidence interval for a parameter $\theta$? Such estimators exist and are called sufficient statistics. An important result is that sufficient statistics can be used to find minimal variance unbiased estimators (MVUE).

### Example 

Suppose we want to construct a confidence interval for the parameter $\lambda$ of an exponential distribution. Let $X_1, X_2, \dots, X_n$ be i.i.d. random variables with $X_i \sim exp(\lambda)$. Construct a $1-\alpha$ confidence interval for $\lambda$. Find a statement $P(L \leq \lambda \leq U) = 1 - \alpha$.

$$
\begin{aligned}
M_X(t) &= (1 - \frac{t}{\lambda})^{-1} \\
M_{\sum_{i=1}^n X_i}(t) &= (1 - \frac{t}{\lambda})^{-n} \\
M_{2\lambda \sum_{i=1}^n X_i}(t) &= (1 - 2t)^{-\frac{2n}{2}}
\end{aligned}
$$

So $2\lambda \sum_{i=1}^n X_i \sim \chi^2_{2n}$

$$
\begin{aligned}
&P\Big( \chi^2_{\frac{\alpha}{2}, 2n} \leq 2\lambda \sum_{i=1}^n X_i \leq \chi^2_{1 - \frac{\alpha}{2}, 2n} \Big) = 1 - \alpha \\
\Rightarrow \quad &P\Big( \frac{\chi^2_{\frac{\alpha}{2}, 2n}}{2\sum_{i=1}^n X_i} \leq \lambda \leq \frac{\chi^2_{1 - \frac{\alpha}{2}, 2n}}{2\sum_{i=1}^n X_i} \Big) = 1 - \alpha
\end{aligned}
$$

We see that to construct the confidence interval, we only need $\sum_{i=1}^n X_i$ (not the individual values $X_1, X_2, \dots, X_n$).

### Sufficiency principle

Let $X_1, X_2, \dots, X_n$ be a random sample and let $T(\mathbf{X}) = T(\mathbf{x})$ be a sufficient statistic.

Let $Y_1, Y_2, \dots, Y_n$ be a random sample and let $T(\mathbf{Y}) = T(\mathbf{y})$ be a sufficient statistic.

If $T(\mathbf{x}) = T(\mathbf{y})$, then the inference we make about the parameter $\theta$ of a distribution will be the same (either using $T(\mathbf{x})$ or $T(\mathbf{y})$).

### Definition (Sufficient statistics)

Let $X_1, X_2, \dots, X_n$ be a random sample and let $T(\mathbf{x})$ be a function of $X_1, X_2, \dots, X_n$. A statistic $T(\mathbf{x})$ is a sufficient statistic for a parameter $\theta$ if the conditional distribution of $\mathbf{X} = \mathbf{X}$ given the value $T(\mathbf{X}) = T(\mathbf{x})$ does not depend on $\theta$.

### Theorem

If $L(\mathbf{X} \mid \theta)$ is the joint probability density or joint probability mass function of $\mathbf{X}$ and $q(t \mid \theta)$ is the pdf or pmf of $T(\mathbf{X})$, we say that $T(\mathbf{X})$ is a sufficient statistic of $\theta$ if the ratio $\frac{L(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)} = H(X_1, X_2, \dots, X_n)$ is a constant as a function of $\theta$.

### Example 

#### (a). Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} \text{Bernoulli}(p)$. Show that $\sum_{i=1}^n X_i$ is a sufficient statistic for $p$.

$$p(x) = p^x(1-p)^{1-x} \qquad T(\mathbf{X}) = \sum_{i=1}^n X_i \sim bin(n,p)$$

$$
\frac{L(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)} = \frac{p^{\sum_{i=1}^n x_i} (1-p)^{n - \sum_{i=1}^n x_i}}{\binom{n}{T(\mathbf{x})} p^{T(\mathbf{x})}(1-p)^{n - T(\mathbf{x})}} = \frac{1}{\binom{n}{T(\mathbf{x})}} \quad \text{(free of } p \text {)}
$$

Since $\frac{L(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)}$ is a constant of $p$, $\sum_{i=1}^n X_i$ is a sufficient statistic for $p$.

#### (b). Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} \Gamma(\alpha,\beta)$. Suppose $\alpha$ is known. Show that $\sum_{i=1}^n X_i$ is a sufficient statistic for $\beta$.

$$f(x) = \frac{x^{\alpha-1} e^{-\frac{x}{\beta}}}{\Gamma(\alpha)\beta^\alpha} \qquad T(\mathbf{X}) = \sum_{i=1}^n X_i \sim \Gamma(n\alpha, \beta)$$

$$
\frac{L(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)} = \frac{ 
\frac{(\prod_{i=1}^n x_i)^{\alpha-1} e^{-\frac{\sum_{i=1}^n x_i}{\beta}}}{\Gamma(\alpha)^n\beta^{n\alpha}} }{ \frac{T(\mathbf{x})^{n\alpha-1} e^{-\frac{T(\mathbf{x})}{\beta}}}{\Gamma(n\alpha)\beta^{n\alpha}} }
= \frac{(\prod_{i=1}^n x_i)^{\alpha-1} \Gamma(n\alpha)}{\Gamma(\alpha)^n T(\mathbf{x})^{n\alpha-1}}
\quad \text{(free of } \beta \text {)}
$$

Since $\frac{L(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)}$ is a constant of $\beta$, $\sum_{i=1}^n X_i$ is a sufficient statistic for $\beta$.

#### (c). Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$ with $\sigma^2$ known. Show that $\bar{X}$ is a sufficient statistic for $\mu$

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2} \qquad T(\mathbf{X}) = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i \sim N(\mu, \frac{\sigma^2}{n})$$

Note:

$$
\sum_{i=1}^n (X_i - \mu)^2
= \sum_{i=1}^n (X_i - \bar{X} + \bar{X} - \mu)^2
= \sum_{i=1}^n (X_i - \bar{X})^2 + n(\bar{X} - \mu)^2 + 2(\bar{X} - \mu) \overbrace{\sum_{i=1}^n (X_i - \bar{X})}^{\text{equal to } 0}
$$

$$
\begin{aligned}
\frac{L(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)} &= \frac{
(2\pi\sigma^2)^{-\frac{n}{2}} e^{-\frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2} }{ 
(2\pi\frac{\sigma^2}{n})^{-\frac{1}{2}} e^{-\frac{1}{2\frac{\sigma^2}{n}} (T(\mathbf{x}) - \mu)^2} } \\
&= \frac{
(2\pi\sigma^2)^{-\frac{n}{2}} e^{-\frac{1}{2\sigma^2} (\sum_{i=1}^n (X_i - \bar{X})^2 + n(\bar{X} - \mu)^2)} }{ (2\pi\frac{\sigma^2}{n})^{-\frac{1}{2}} e^{-\frac{1}{2\sigma^2} n(T(\mathbf{x}) - \mu)^2} } \\
&= \frac{ (2\pi\sigma^2)^{-\frac{n}{2}} }{ (2\pi\frac{\sigma^2}{n})^{-\frac{1}{2}} } e^{-\frac{1}{2\sigma^2} \sum_{i=1}^n (X_i - \bar{X})^2}
\qquad \text{(free of } \mu \text {)}
\end{aligned}
$$

Since $\frac{L(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)}$ is a constant of $\mu$, $\bar{X}$ is a sufficient statistic for $\mu$.

#### (d). Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} \text{Poisson}(\lambda)$. Show that $\sum_{i=1}^n X_i$ a sufficient statistic for $\lambda$.

$$f(x) = \frac{\lambda^x e^{-\lambda}}{x!} \qquad T(\mathbf{X}) = \sum_{i=1}^n X_i \sim \text{Poisson}(n\lambda)$$

$$
\frac{L(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)} = \frac{ 
\frac{\lambda^{\sum_{i=1}^n x_i} e^{-n\lambda}}{\prod_{i=1}^nx!} }{
\frac{(n\lambda)^{T(\mathbf{x})} e^{-n\lambda}}{T(\mathbf{x})!} } =
\frac{T(\mathbf{x})!}{n^{T(\mathbf{x})} \prod_{i=1}^nx!}
\qquad \text{(free of } \lambda \text {)}
$$

Since $\frac{L(\mathbf{x} \mid \theta)}{q(T(\mathbf{x}) \mid \theta)}$ is a constant of $\lambda$, $\sum_{i=1}^n X_i$ is a sufficient statistic for $\lambda$.

### Factorization theorem

Let $X_1, X_2, \dots, X_n$ be a random sample and let $L(\mathbf{x}; \theta)$ be the likelihood function. The $T(\mathbf{x})$ is a sufficient statistic for the estimation of a parameter $\theta$ iff the likelihood function can be expressed as the product of two non-negative functions:

$$L(\mathbf{x}; \theta) = g(T(\mathbf{x}); \theta) h(\mathbf{x})$$

### Example 

#### (a). Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$ with $\sigma^2$ known. Show that $\bar{X}$ is a sufficient statistic for $\mu$ (revisit):

$$L(\mathbf{x}; \theta) 
= (2\pi\sigma^2)^{-\frac{n}{2}} e^{-\frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2} 
= \underbrace{(2\pi\sigma^2)^{-\frac{n}{2}} e^{-\frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \bar{x})^2}}_{h(\mathbf{x})} \underbrace{e^{-\frac{n}{2\sigma^2} (\bar{x} - \mu)^2 }}_{g(T(\mathbf{x}); \mu)}$$

Therefore, $\bar{X}$ is a sufficient statistic of $\mu$.

#### (b). Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} exp(\frac{1}{\lambda})$. Show that $T(\mathbf{X}) = \sum_{i=1}^n X_i$ is a sufficient statistic for the estimation of $\lambda$.

$$L(\mathbf{x}; \theta) = \frac{1}{\lambda^n} e^{-\frac{1}{\lambda} \sum_{i=1}^n x_i} \quad \text{so } h(x) = 1$$

Therefore, $\sum_{i=1}^n X_i$ is a sufficient statistic of $\lambda$.

#### (c). Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$ with both $\mu$ and $\sigma^2$ unknown. Show that $(\bar{X}, s^2)$ are sufficient statistics for $(\mu, \sigma^2)$.

$$
\begin{aligned}
L(\mathbf{x}; \theta) 
&= (2\pi\sigma^2)^{-\frac{n}{2}} e^{-\frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2} \\
&= (2\pi\sigma^2)^{-\frac{n}{2}} e^{-\frac{1}{2\sigma^2} (\sum_{i=1}^n (x_i - \bar{x})^2 + n(\bar{x} - \mu)^2)} \\
&= (2\pi\sigma^2)^{-\frac{n}{2}} e^{-\frac{1}{2\sigma^2} ((n-1)s^2 + n(\bar{x} - \mu)^2)} \qquad \text{where } s^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n-1} \text{ and } h(\mathbf{x}) = 1
\end{aligned}
$$

Therefore, $(\bar{X}, s^2)$ are sufficient statistics for $(\mu, \sigma^2)$.

### Properties of sufficient statistics

#### 1. Functions of sufficient statistics are also sufficient.

##### Example - Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} N(0, \sigma^2)$. Show that $\frac{1}{n} \sum_{i=1}^n X_i^2$ (second sample moment) is a sufficient statistic for $\sigma^2$.

$$
L(\mathbf{x}; \theta) 
= (2\pi\sigma^2)^{-\frac{n}{2}} e^{-\frac{1}{2\sigma^2} \sum_{i=1}^n x_i^2} 
= (2\pi\sigma^2)^{-\frac{n}{2}} e^{-\frac{1}{2\sigma^2} n \frac{1}{n}\sum_{i=1}^n x_i^2}
$$

Therefore, $\frac{1}{n} \sum_{i=1}^n X_i^2$ is a sufficient statistic of $\sigma^2$.

##### Example - Let $X_1, X_2, \dots, X_n \overset{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$. Show that $(\sum_{i=1}^n X_i, \sum_{i=1}^n X_i^2)$ are sufficient statistics for $(\mu, \sigma^2)$.

Previously have shown that $(\bar{X}, s^2)$ are sufficient statistics for $(\mu, \sigma^2)$.

$s^2 = \frac{\sum_{i=1}^n (X_i - \bar{X})^2}{n-1} = \frac{\sum_{i=1}^n X_i^2 - n\bar{X}^2}{n-1} \quad \Rightarrow \quad \sum_{i=1}^n X_i^2 = (n-1)s^2 + n\bar{X}^2$.

Thus $(\sum_{i=1}^n X_i, \sum_{i=1}^n X_i^2)$ are sufficient statistics for $(\mu, \sigma^2)$.

#### 2. The maximum likelihood estimates are functions of sufficient statistics.

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

Use factorization theorem: $L(\mathbf{x}; \theta) = g(T(\mathbf{x}); \theta) h(\mathbf{x})$

Since $h(\mathbf{x})$ does not depend on $\theta$, maximizing the likelihood with respect to $\theta$ is equivalent to maximizing $g(T(\mathbf{x});\theta)$.

Hence, 

$$\hat{\theta} = \arg\max_{\theta} L(\mathbf{x};\theta) = \arg\max_{\theta} g(T(\mathbf{x});\theta)$$ 

which depends on the sample only through the sufficient statistic $T(\mathbf{x})$. Therefore, the maximum likelihood estimator is a function of the sufficient statistic.

</div>

#### 3. Let $X_1, X_2, \dots, X_n$ be i.i.d. random variables from a pdf or pmf that belongs in an exponential family:

$$f(x \mid \boldsymbol{\theta}) = h(x)c(\boldsymbol{\theta}) \exp\Big(\sum_{i=1}^k w_i(\boldsymbol{\theta})t_i(x)\Big)$$

Then $T(\mathbf{X}) = \Big( \sum_{j=1}^n t_1(X_j), \dots, \sum_{j=1}^n t_k(X_j) \Big)$ is a sufficient statistic of $\boldsymbol{\theta}$.

<div style="border:2px solid black; padding:12px; margin:12px 0; border-radius:4px;">
    
**Proof**:

