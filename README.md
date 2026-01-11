

# **Structural Estimation of Consumer Demand using the BLP Model**

This project implements the **Berry–Levinsohn–Pakes (BLP) random coefficients logit model** to structurally estimate consumer demand in differentiated product markets. The implementation follows the classical BLP framework with **consumer heterogeneity, contraction mapping for market share inversion, and GMM estimation**.

The model is implemented from scratch in Python using Jupyter Notebook.

---

## **Project Overview**

The BLP model extends the standard logit model by allowing **heterogeneous consumer preferences** over product characteristics. It recovers mean utilities that rationalize observed market shares and estimates preference parameters using instrumental variables to address price endogeneity.

This project includes:

* Utility decomposition with random coefficients
* Simulation of consumer heterogeneity
* Market share inversion using contraction mapping
* Structural error recovery
* GMM-based parameter estimation
* Elasticity computation and model validation

---

## **Model Specification**

### **Consumer Utility**

For consumer *i* and product *j*:

[
U_{ij} = x_j' \theta_i + \delta_j + \varepsilon_{ij}
]

Where:

* ( x_j ): product characteristics (price, features, etc.)
* ( \theta_i ): individual-specific taste coefficients
* ( \delta_j ): mean utility
* ( \varepsilon_{ij} ): i.i.d. Type-I Extreme Value error

Individual coefficients are modeled as:

[
\theta_i = \bar{\theta} + \Pi D_i + \Sigma \nu_i
]

This captures both **observed (demographics)** and **unobserved heterogeneity** 

---

## **Estimation Strategy**

### **1. Market Share Inversion (Contraction Mapping)**

For a given parameter vector, mean utilities ( \delta ) are recovered by solving:

[
\delta^{(t+1)} = \delta^{(t)} + \log(s^{obs}) - \log(\hat{s}(\delta^{(t)}, \theta))
]

This mapping is a contraction and converges to the unique solution that matches observed market shares 

---

### **2. Structural Error Decomposition**

Mean utility is decomposed as:

[
\delta_j = x_j' \bar{\theta} + \xi_j
]

where ( \xi_j ) is the **unobserved product quality shock** 

---

### **3. GMM Estimation**

Because price is endogenous, the model uses instrumental variables ( Z ) and estimates parameters by minimizing:

[
Q(\theta) = g(\theta)' W g(\theta), \quad g(\theta) = \frac{1}{J} Z' \xi(\theta)
]

A two-step GMM procedure is implemented with optimal weighting matrix 

---

## **Implementation Structure**

Your Jupyter Notebook follows this pipeline:

1. **Data Preparation**

   * Product characteristics matrix ( X )
   * Observed market shares ( s_{obs} )
   * Instrument matrix ( Z )

2. **Consumer Simulation**

   * Draws ( \nu_i \sim N(0, I) )
   * Optional demographics ( D_i )

3. **Utility Computation**

   * Individual coefficients ( \theta_i )
   * Utility matrix ( V_{ij} )

4. **Choice Probabilities**

   * Logit probabilities with numerical stabilization

5. **Contraction Mapping**

   * Iterative recovery of ( \delta )

6. **Structural Error Recovery**

   * ( \xi = \delta - X \bar{\theta} )

7. **GMM Objective Evaluation**

   * Moment computation and minimization

8. **Elasticity Computation**

   * Own-price and cross-price elasticities

---

## **Key Functions Implemented**

* `simulate_consumers()` – generate heterogeneity draws
* `compute_individual_coefficients()` – build θᵢ
* `compute_utilities()` – calculate Vᵢⱼ
* `choice_probabilities()` – multinomial logit probabilities
* `aggregate_market_shares()` – predicted shares
* `blp_contraction_mapping()` – invert market shares
* `compute_xi()` – structural errors
* `gmm_objective()` – GMM loss function
* `estimate_parameters()` – optimizer wrapper
* `compute_elasticities()` – elasticity matrix

(All consistent with your documentation structure )

---

## **Model Validation**

The implementation includes:

* Convergence checks for contraction mapping
* Overidentification (Hansen J) test
* Predicted vs observed share comparison
* Elasticity sign and magnitude validation 

---

## **Example Use Case**

The framework is suitable for:

* Automobile markets
* Consumer goods
* Differentiated product industries
* Merger analysis and counterfactual simulations

---

## **Technical Skills Demonstrated**

* Advanced microeconometrics (BLP, GMM, IV)
* Numerical optimization
* Simulation-based integration
* High-dimensional matrix computation
* Econometric identification and inference

---

## **References**

Berry, Levinsohn & Pakes (1995), *Econometrica*
Nevo (2000, 2001), *Econometrica*
Train (2009), *Discrete Choice Methods with Simulation*
Conlon & Gortmaker (2020), *RAND Journal of Economics* 

---

## **Author**

Himanshu Kushwaha
BS Economics, IIT Patna
Interests: Quantitative Finance, Econometrics, Data Science

