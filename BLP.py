import numpy as np

class UtilityEstimation:
    """
    Utility estimation with simulated consumers (BLP-style).
    
    This class implements the BLP (Berry-Levinsohn-Pakes) inner loop:
    given nonlinear parameters (theta_1, pi, sigma), it solves for the 
    mean utilities delta using contraction mapping.
    
    Model:
        U_ij = x_j' θ_i + δ_j + ε_ij
        θ_i = θ̄ + Π D_i + Σ v_i
    
    Where:
        - U_ij: utility of consumer i for product j
        - x_j: product characteristics (K x 1)
        - θ_i: individual-specific coefficients
        - δ_j: mean utility (to be recovered)
        - ε_ij: i.i.d. Type-I extreme value error
        - D_i: demographics (D x 1)
        - v_i: unobserved taste shocks (K x 1)
        - θ̄ (theta_1): mean coefficients (K x 1)
        - Π (pi): demographic interactions (K x D)
        - Σ (sigma): random coefficients std dev (K x 1)
    """
    
    # -------------------------------------------------- 
    # Initialization
    # --------------------------------------------------
    def __init__(self, X, s_jt, D=None, ns=500, seed=42):
        """
        Initialize the BLP estimation framework.
        
        Parameters:
        -----------
        X : np.ndarray, shape (J, K)
            Product characteristics matrix
            - J: number of products
            - K: number of product characteristics
            
        s_jt : np.ndarray, shape (J,)
            Observed market shares for each product
            
        D : np.ndarray, shape (ns, D_dim) or None
            Demographics for simulated consumers
            - ns: number of simulated consumers
            - D_dim: number of demographic variables
            If None, no demographics used
            
        ns : int, default=500
            Number of simulated consumers for integration
            
        seed : int, default=42
            Random seed for reproducibility
        """
        self.X = X  # (J, K)
        self.s_jt = s_jt  # (J,)
        self.J = X.shape[0]  # number of products
        self.K = X.shape[1]  # number of characteristics
        self.ns = ns  # number of simulated consumers
        self.D = D  # demographics (ns, D_dim) or None
        self.D_dim = D.shape[1] if D is not None else 0
        
        np.random.seed(seed)
        self._simulate_consumers()
    
    def _simulate_consumers(self):
        """
        Simulate consumer heterogeneity draws.
        
        Creates:
        --------
        self.nu : np.ndarray, shape (ns, K)
            Standard normal draws for random coefficients
            Each row represents one simulated consumer's taste shock vector
        """
        self.nu = np.random.randn(self.ns, self.K)  # (ns, K)
    
    def compute_individual_coefficients(self, theta_1, pi=None, sigma=None):
        """
        Compute individual-specific coefficients θ_i for all consumers.
        
        θ_i = θ̄ + Π * D_i + Σ * v_i
        
        Parameters:
        -----------
        theta_1 : np.ndarray, shape (K,)
            Mean taste coefficients θ̄
            
        pi : np.ndarray, shape (K, D_dim) or None
            Demographic interaction coefficients Π
            Maps demographics to taste heterogeneity
            If None, no demographic interactions
            
        sigma : np.ndarray, shape (K,) or None
            Standard deviations for random coefficients Σ
            Controls unobserved heterogeneity
            If None, no random coefficients (pure logit)
        
        Returns:
        --------
        coeffs : np.ndarray, shape (ns, K)
            Individual-specific coefficients for each consumer
            Row i contains the K coefficients for consumer i
        """
        coeffs = np.tile(theta_1, (self.ns, 1))  # (ns, K) - broadcast mean
        
        # Add demographic interactions: Π * D_i
        if pi is not None and self.D is not None:
            coeffs += self.D @ pi.T  # (ns, D_dim) @ (D_dim, K) -> (ns, K)
        
        # Add random coefficient variation: Σ * v_i
        if sigma is not None:
            coeffs += self.nu * sigma[np.newaxis, :]  # (ns, K) * (1, K)
        
        return coeffs  # (ns, K)
    
    # --------------------------------------------------
    # Utility computation
    # --------------------------------------------------
    def compute_utilities(self, coeffs, delta):
        """
        Compute indirect utilities for all consumers and products.
        
        V_ij = x_j' * θ_i + δ_j
        
        Parameters:
        -----------
        coeffs : np.ndarray, shape (ns, K)
            Individual-specific coefficients θ_i
            
        delta : np.ndarray, shape (J,)
            Mean utilities δ_j for each product
        
        Returns:
        --------
        V : np.ndarray, shape (ns, J)
            Utility matrix where V[i,j] is utility of consumer i for product j
            Does NOT include ε_ij (idiosyncratic error)
        """
        # X: (J, K), coeffs.T: (K, ns) -> (J, ns) -> transpose to (ns, J)
        V = (self.X @ coeffs.T).T + delta[np.newaxis, :]  # (ns, J)
        return V
    
    # --------------------------------------------------
    # Choice probabilities (logit with outside good)
    # --------------------------------------------------
    def choice_probabilities(self, V):
        """
        Compute individual choice probabilities using logit formula.
        
        P_ij = exp(V_ij) / (1 + Σ_k exp(V_ik))
        
        Includes outside good (normalized utility = 0).
        
        Parameters:
        -----------
        V : np.ndarray, shape (ns, J)
            Utility matrix (before idiosyncratic shocks)
        
        Returns:
        --------
        P : np.ndarray, shape (ns, J)
            Choice probability matrix where P[i,j] is probability 
            consumer i chooses product j
        """
        exp_V = np.exp(V - np.max(V, axis=1, keepdims=True))  # numerical stability
        denom = 1 + np.sum(exp_V, axis=1, keepdims=True)  # (ns, 1)
        P = exp_V / denom  # (ns, J)
        return P
    
    # --------------------------------------------------
    # Aggregate market shares
    # --------------------------------------------------
    def aggregate_market_shares(self, P):
        """
        Aggregate individual probabilities to market shares.
        
        s_j = (1/ns) * Σ_i P_ij
        
        Parameters:
        -----------
        P : np.ndarray, shape (ns, J)
            Individual choice probabilities
        
        Returns:
        --------
        s_pred : np.ndarray, shape (J,)
            Predicted market shares by averaging over simulated consumers
        """
        s_pred = np.mean(P, axis=0)  # (J,)
        return s_pred
    
    # --------------------------------------------------
    # Initial delta (simple logit inversion)
    # --------------------------------------------------
    def compute_initial_delta(self):
        """
        Compute initial guess for δ using simple logit inversion.
        
        δ_j = ln(s_j) - ln(s_0)
        
        where s_0 is the outside good share (1 - Σ s_j).
        
        Returns:
        --------
        delta_init : np.ndarray, shape (J,)
            Initial mean utilities based on observed shares
            
        Notes:
        ------
        This is the closed-form solution for the pure logit model
        (no random coefficients or demographics).
        """
        s_0 = 1 - np.sum(self.s_jt)  # outside good share
        if s_0 <= 0:
            raise ValueError("Market shares sum to ≥1. Outside good share must be positive.")
        if np.any(self.s_jt <= 0):
            raise ValueError("All observed shares must be positive for log transformation.")
        delta_init = np.log(self.s_jt) - np.log(s_0)  # (J,)
        return delta_init
    
    # --------------------------------------------------
    # BLP contraction mapping
    # --------------------------------------------------
    def blp_contraction_mapping(self, theta_1, pi=None, sigma=None, 
                                max_iter=1000, tol=1e-12):
        """
        Solve for mean utilities δ using contraction mapping.
        
        Iteration: δ^(t+1) = δ^(t) + ln(s_observed) - ln(s_predicted(δ^(t)))
        
        Parameters:
        -----------
        theta_1 : np.ndarray, shape (K,)
            Mean taste coefficients
            
        pi : np.ndarray, shape (K, D_dim) or None
            Demographic interactions
            
        sigma : np.ndarray, shape (K,) or None
            Random coefficient standard deviations
            
        max_iter : int, default=1000
            Maximum number of contraction iterations
            
        tol : float, default=1e-12
            Convergence tolerance (max absolute difference in δ)
        
        Returns:
        --------
        delta : np.ndarray, shape (J,)
            Converged mean utilities
            
        converged : bool
            Whether the contraction converged within max_iter
        """
        # Initialize with logit inversion
        delta = self.compute_initial_delta()  # (J,)
        
        # Compute individual coefficients (fixed for this θ)
        coeffs = self.compute_individual_coefficients(theta_1, pi, sigma)  # (ns, K)
        
        for iteration in range(max_iter):
            # Compute utilities and choice probabilities
            V = self.compute_utilities(coeffs, delta)  # (ns, J)
            P = self.choice_probabilities(V)  # (ns, J)
            
            # Aggregate to market shares
            s_pred = self.aggregate_market_shares(P)  # (J,)
            
            # Contraction update
            delta_new = delta + np.log(self.s_jt) - np.log(s_pred)  # (J,)
            
            # Check convergence
            if self.check_convergence(delta, delta_new, tol):
                return delta_new, True
            
            delta = delta_new
        
        return delta, False
    
    def compute_xi(self, delta, X, theta_1):
        """
        Compute structural error (unobserved product quality).
        
        ξ_j = δ_j - x_j' * θ̄
        
        Parameters:
        -----------
        delta : np.ndarray, shape (J,)
            Mean utilities from contraction mapping
            
        X : np.ndarray, shape (J, K)
            Product characteristics
            
        theta_1 : np.ndarray, shape (K,)
            Mean taste coefficients
        
        Returns:
        --------
        xi : np.ndarray, shape (J,)
            Structural errors (unobserved quality shocks)
            
        Notes:
        ------
        ξ captures quality/features not observed in X.
        In valid models, E[Z' ξ] = 0 for instruments Z.
        """
        xi = delta - X @ theta_1  # (J,)
        return xi
    
    def compute_moments(self, xi, Z):
        """
        Compute moment conditions for GMM.
        
        g(θ) = Z' * ξ(θ)
        
        Parameters:
        -----------
        xi : np.ndarray, shape (J,)
            Structural errors
            
        Z : np.ndarray, shape (J, L)
            Instruments
            - L: number of instruments
        
        Returns:
        --------
        moments : np.ndarray, shape (L,)
            Sample moment conditions
            
        Notes:
        ------
        In a correctly specified model with valid instruments,
        E[g(θ_0)] = 0 at the true parameter θ_0.
        """
        moments = Z.T @ xi  # (L, J) @ (J,) -> (L,)
        return moments / self.J  # normalize by sample size
    
    def gmm_objective(self, theta_params, Z, W=None):
        """
        GMM objective function to minimize.
        
        Q(θ) = g(θ)' * W * g(θ)
        
        Parameters:
        -----------
        theta_params : np.ndarray, shape (n_params,)
            All nonlinear parameters stacked:
            [theta_1 (K,), vec(pi) (K*D_dim,), sigma (K,)]
            
        Z : np.ndarray, shape (J, L)
            Instruments
            
        W : np.ndarray, shape (L, L) or None
            Weighting matrix for GMM
            If None, uses identity (equal weighting)
        
        Returns:
        --------
        obj : float
            GMM objective value (scalar)
            
        Notes:
        ------
        Optimal W = (Z'Z)^(-1) or uses two-step/iterated GMM.
        """
        # Unpack parameters
        theta_1 = theta_params[:self.K]  # (K,)
        
        if self.D_dim > 0:
            pi_vec = theta_params[self.K:self.K + self.K * self.D_dim]
            pi = pi_vec.reshape(self.K, self.D_dim)  # (K, D_dim)
            sigma = theta_params[self.K + self.K * self.D_dim:]  # (K,)
        else:
            pi = None
            sigma = theta_params[self.K:]  # (K,)
        
        # Solve for delta via contraction
        delta, converged = self.blp_contraction_mapping(theta_1, pi, sigma)
        
        if not converged:
            return 1e10  # penalty for non-convergence
        
        # Compute structural error
        xi = self.compute_xi(delta, self.X, theta_1)  # (J,)
        
        # Compute moments
        g = self.compute_moments(xi, Z)  # (L,)
        
        # GMM objective
        if W is None:
            W = np.eye(Z.shape[1])  # (L, L)
        
        obj = g.T @ W @ g  # scalar
        return obj
    
    def estimate_parameters(self, initial_theta, Z, method='Nelder-Mead', 
                          maxiter=1000):
        """
        Full BLP estimation routine using GMM.
        
        Parameters:
        -----------
        initial_theta : np.ndarray, shape (n_params,)
            Initial parameter guess
            Format: [theta_1, vec(pi), sigma]
            
        Z : np.ndarray, shape (J, L)
            Instrumental variables
            
        method : str, default='Nelder-Mead'
            Optimization algorithm (passed to scipy.optimize.minimize)
            
        maxiter : int, default=1000
            Maximum optimization iterations
        
        Returns:
        --------
        result : dict
            Estimation results containing:
            - 'theta': estimated parameters (n_params,)
            - 'delta': mean utilities at optimum (J,)
            - 'xi': structural errors at optimum (J,)
            - 'objective': final GMM objective value
            - 'success': whether optimization converged
            
        Notes:
        ------
        This is a simplified version. Production code should:
        - Implement two-step GMM with optimal weighting
        - Compute standard errors (via delta method or bootstrap)
        - Add parameter bounds and constraints
        """
        from scipy.optimize import minimize
        
        # First-stage: equal weighting
        result = minimize(
            lambda theta: self.gmm_objective(theta, Z, W=None),
            initial_theta,
            method=method,
            options={'maxiter': maxiter}
        )
        
        # Unpack final parameters
        theta_opt = result.x
        theta_1 = theta_opt[:self.K]
        
        if self.D_dim > 0:
            pi_vec = theta_opt[self.K:self.K + self.K * self.D_dim]
            pi = pi_vec.reshape(self.K, self.D_dim)
            sigma = theta_opt[self.K + self.K * self.D_dim:]
        else:
            pi = None
            sigma = theta_opt[self.K:]
        
        # Recover delta and xi at optimum
        delta_final, _ = self.blp_contraction_mapping(theta_1, pi, sigma)
        xi_final = self.compute_xi(delta_final, self.X, theta_1)
        
        return {
            'theta': theta_opt,
            'delta': delta_final,
            'xi': xi_final,
            'objective': result.fun,
            'success': result.success
        }
    
    def compute_elasticities(self, theta_1, pi=None, sigma=None, 
                           delta=None, price_idx=0):
        """
        Compute own- and cross-price elasticities.
        
        ε_jk = (∂s_j / ∂p_k) * (p_k / s_j)
        
        Parameters:
        -----------
        theta_1 : np.ndarray, shape (K,)
            Mean taste coefficients
            
        pi : np.ndarray, shape (K, D_dim) or None
            Demographic interactions
            
        sigma : np.ndarray, shape (K,) or None
            Random coefficient standard deviations
            
        delta : np.ndarray, shape (J,) or None
            Mean utilities (if None, solves via contraction)
            
        price_idx : int, default=0
            Index of price variable in X matrix
        
        Returns:
        --------
        elasticities : np.ndarray, shape (J, J)
            Elasticity matrix where element (j,k) is elasticity of 
            product j's share with respect to product k's price
            Diagonal: own-price elasticities (negative)
            Off-diagonal: cross-price elasticities
            
        Notes:
        ------
        Formula:
        - Own: ε_jj = -α * p_j * (1 - s_j)
        - Cross: ε_jk = α * p_k * s_k
        where α is the price coefficient (could be random)
        """
        # Solve for delta if not provided
        if delta is None:
            delta, _ = self.blp_contraction_mapping(theta_1, pi, sigma)
        
        # Compute individual coefficients and probabilities
        coeffs = self.compute_individual_coefficients(theta_1, pi, sigma)
        V = self.compute_utilities(coeffs, delta)
        P = self.choice_probabilities(V)  # (ns, J)
        s = self.aggregate_market_shares(P)  # (J,)
        
        # Extract prices
        prices = self.X[:, price_idx]  # (J,)
        
        # Price coefficients for each consumer
        alpha_i = coeffs[:, price_idx]  # (ns,)
        
        # Compute elasticities
        elasticities = np.zeros((self.J, self.J))
        
        for j in range(self.J):
            for k in range(self.J):
                if j == k:
                    # Own-price elasticity
                    # ε_jj = -(1/s_j) * p_j * E[α_i * P_ij * (1 - P_ij)]
                    term = np.mean(alpha_i * P[:, j] * (1 - P[:, j]))
                    elasticities[j, j] = -prices[j] * term / s[j]
                else:
                    # Cross-price elasticity
                    # ε_jk = -(1/s_j) * p_k * E[α_i * P_ij * P_ik]
                    term = np.mean(alpha_i * P[:, j] * P[:, k])
                    elasticities[j, k] = prices[k] * term / s[j]
        
        return elasticities
    
    def check_convergence(self, delta_old, delta_new, tol):
        """
        Check convergence of contraction mapping.
        
        Parameters:
        -----------
        delta_old : np.ndarray, shape (J,)
            Previous iteration values
            
        delta_new : np.ndarray, shape (J,)
            Current iteration values
            
        tol : float
            Convergence tolerance
        
        Returns:
        --------
        converged : bool
            True if max|delta_new - delta_old| < tol
        """
        return np.max(np.abs(delta_new - delta_old)) < tol