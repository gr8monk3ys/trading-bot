"""
Factor Orthogonalization and Risk Parity Weighting

Addresses two key issues with naive factor combination:
1. Factor correlation - Momentum and sector-relative momentum are correlated,
   diluting alpha. Orthogonalization removes redundancy.
2. Unequal risk contribution - A 10% weight in high-vol factor may contribute
   50% of portfolio risk. Risk parity equalizes risk contribution.

Methods:
- PCA orthogonalization: Extract uncorrelated principal components
- Gram-Schmidt: Sequential orthogonalization preserving factor interpretation
- Risk parity: Weight factors so each contributes equal variance

Usage:
    ortho = FactorOrthogonalizer(factor_scores, method='pca')
    orthogonal_scores = ortho.orthogonalize()

    rp = RiskParityWeighter(factor_returns)
    weights = rp.calculate_weights()
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class OrthogonalizationMethod(Enum):
    """Methods for factor orthogonalization."""
    PCA = "pca"  # Principal Component Analysis
    GRAM_SCHMIDT = "gram_schmidt"  # Sequential orthogonalization
    SYMMETRIC = "symmetric"  # Symmetric orthogonalization (preserves scale)


@dataclass
class OrthogonalizedFactors:
    """Result of factor orthogonalization."""

    original_factors: List[str]
    orthogonal_scores: Dict[str, Dict[str, float]]  # factor -> {symbol -> score}
    transformation_matrix: np.ndarray  # Maps original to orthogonal
    explained_variance_ratio: List[float]  # For PCA
    factor_correlations_before: np.ndarray
    factor_correlations_after: np.ndarray
    method: OrthogonalizationMethod
    timestamp: datetime

    @property
    def correlation_reduction(self) -> float:
        """Measure of correlation removed (1.0 = perfect orthogonalization)."""
        # Compare off-diagonal elements
        n = len(self.original_factors)
        if n < 2:
            return 0.0

        before_offdiag = np.abs(self.factor_correlations_before[np.triu_indices(n, 1)])
        after_offdiag = np.abs(self.factor_correlations_after[np.triu_indices(n, 1)])

        if before_offdiag.mean() == 0:
            return 0.0

        return 1.0 - (after_offdiag.mean() / before_offdiag.mean())


@dataclass
class RiskParityWeights:
    """Result of risk parity weighting."""

    factor_weights: Dict[str, float]
    risk_contributions: Dict[str, float]  # Each should be ~equal
    factor_volatilities: Dict[str, float]
    factor_correlations: np.ndarray
    optimization_converged: bool
    timestamp: datetime

    @property
    def risk_concentration(self) -> float:
        """Herfindahl index of risk contributions (0 = perfect parity)."""
        contribs = list(self.risk_contributions.values())
        if not contribs:
            return 0.0
        total = sum(contribs)
        if total == 0:
            return 0.0
        normalized = [c / total for c in contribs]
        return sum(c ** 2 for c in normalized) - (1.0 / len(contribs))


class FactorOrthogonalizer:
    """
    Removes correlations between factors via orthogonalization.

    Why this matters:
    - Momentum (12-month) correlates ~0.7 with sector-relative momentum
    - Value (P/E) correlates ~0.6 with quality (ROE for profitable firms)
    - Correlated factors don't provide diversification benefits
    - Orthogonalization extracts unique signal from each factor

    PCA advantages:
    - Statistically optimal (maximizes explained variance)
    - Easy to interpret (PC1 = largest common driver)

    Gram-Schmidt advantages:
    - Preserves interpretation of first factors
    - Controllable ordering (put most important factor first)
    """

    def __init__(
        self,
        factor_scores: Dict[str, Dict[str, float]],
        method: OrthogonalizationMethod = OrthogonalizationMethod.PCA,
        min_observations: int = 20,
    ):
        """
        Initialize orthogonalizer.

        Args:
            factor_scores: Dict of factor_name -> {symbol -> score}
            method: Orthogonalization method
            min_observations: Minimum symbols required
        """
        self.factor_scores = factor_scores
        self.method = method
        self.min_observations = min_observations

        # Build factor matrix
        self.factor_names = list(factor_scores.keys())
        self.symbols = self._get_common_symbols()
        self.factor_matrix = self._build_factor_matrix()

    def _get_common_symbols(self) -> List[str]:
        """Get symbols present in all factors."""
        if not self.factor_scores:
            return []

        symbol_sets = [set(scores.keys()) for scores in self.factor_scores.values()]
        common = set.intersection(*symbol_sets) if symbol_sets else set()
        return sorted(common)

    def _build_factor_matrix(self) -> np.ndarray:
        """Build matrix of factor scores (symbols x factors)."""
        if not self.symbols or not self.factor_names:
            return np.array([])

        matrix = np.zeros((len(self.symbols), len(self.factor_names)))

        for j, factor_name in enumerate(self.factor_names):
            for i, symbol in enumerate(self.symbols):
                matrix[i, j] = self.factor_scores[factor_name].get(symbol, np.nan)

        return matrix

    def orthogonalize(self) -> Optional[OrthogonalizedFactors]:
        """
        Perform factor orthogonalization.

        Returns:
            OrthogonalizedFactors or None if insufficient data
        """
        if len(self.symbols) < self.min_observations:
            logger.warning(
                f"Insufficient symbols for orthogonalization: "
                f"{len(self.symbols)} < {self.min_observations}"
            )
            return None

        if len(self.factor_names) < 2:
            logger.warning("Need at least 2 factors for orthogonalization")
            return None

        # Handle missing values
        valid_mask = ~np.isnan(self.factor_matrix).any(axis=1)
        valid_matrix = self.factor_matrix[valid_mask]
        valid_symbols = [s for s, v in zip(self.symbols, valid_mask, strict=False) if v]

        if len(valid_symbols) < self.min_observations:
            logger.warning(f"Insufficient valid observations: {len(valid_symbols)}")
            return None

        # Standardize factors (zero mean, unit variance)
        means = np.mean(valid_matrix, axis=0)
        stds = np.std(valid_matrix, axis=0)
        stds[stds == 0] = 1  # Avoid division by zero
        standardized = (valid_matrix - means) / stds

        # Calculate correlation before
        corr_before = np.corrcoef(standardized.T)

        # Orthogonalize
        if self.method == OrthogonalizationMethod.PCA:
            orthogonal, transform, explained = self._pca_orthogonalize(standardized)
        elif self.method == OrthogonalizationMethod.GRAM_SCHMIDT:
            orthogonal, transform, explained = self._gram_schmidt_orthogonalize(standardized)
        else:  # SYMMETRIC
            orthogonal, transform, explained = self._symmetric_orthogonalize(standardized)

        # Calculate correlation after
        corr_after = np.corrcoef(orthogonal.T)

        # Build orthogonal factor scores dict
        ortho_scores = {}
        for j, factor_name in enumerate(self.factor_names):
            ortho_name = f"{factor_name}_ortho"
            ortho_scores[ortho_name] = {
                valid_symbols[i]: float(orthogonal[i, j])
                for i in range(len(valid_symbols))
            }

        result = OrthogonalizedFactors(
            original_factors=self.factor_names,
            orthogonal_scores=ortho_scores,
            transformation_matrix=transform,
            explained_variance_ratio=explained,
            factor_correlations_before=corr_before,
            factor_correlations_after=corr_after,
            method=self.method,
            timestamp=datetime.now(),
        )

        logger.info(
            f"Orthogonalized {len(self.factor_names)} factors using {self.method.value}. "
            f"Correlation reduction: {result.correlation_reduction:.1%}"
        )

        return result

    def _pca_orthogonalize(
        self, matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        PCA orthogonalization.

        Returns:
            (orthogonal_matrix, transformation_matrix, explained_variance_ratio)
        """
        # Compute covariance matrix
        cov = np.cov(matrix.T)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Transform to PC space
        orthogonal = matrix @ eigenvectors

        # Explained variance ratio
        total_var = np.sum(eigenvalues)
        explained = (eigenvalues / total_var).tolist() if total_var > 0 else []

        return orthogonal, eigenvectors, explained

    def _gram_schmidt_orthogonalize(
        self, matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Gram-Schmidt orthogonalization.

        Preserves interpretation of first factor, then orthogonalizes subsequent
        factors with respect to earlier ones.

        Returns:
            (orthogonal_matrix, transformation_matrix, [])
        """
        n_factors = matrix.shape[1]
        orthogonal = np.zeros_like(matrix)
        transform = np.eye(n_factors)

        for j in range(n_factors):
            # Start with original factor
            orthogonal[:, j] = matrix[:, j].copy()

            # Subtract projections onto previous orthogonal factors
            for k in range(j):
                if np.linalg.norm(orthogonal[:, k]) > 0:
                    proj_coef = (
                        np.dot(orthogonal[:, j], orthogonal[:, k]) /
                        np.dot(orthogonal[:, k], orthogonal[:, k])
                    )
                    orthogonal[:, j] -= proj_coef * orthogonal[:, k]
                    transform[k, j] = -proj_coef

            # Normalize
            norm = np.linalg.norm(orthogonal[:, j])
            if norm > 0:
                orthogonal[:, j] /= norm

        return orthogonal, transform, []

    def _symmetric_orthogonalize(
        self, matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Symmetric (Löwdin) orthogonalization.

        Treats all factors equally (no ordering preference).
        Minimizes deviation from original factors.

        Returns:
            (orthogonal_matrix, transformation_matrix, [])
        """
        # Correlation matrix
        corr = np.corrcoef(matrix.T)

        # Symmetric square root inverse
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability

        sqrt_inv = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        # Transform
        orthogonal = matrix @ sqrt_inv

        return orthogonal, sqrt_inv, []


class RiskParityWeighter:
    """
    Calculates factor weights using risk parity principle.

    Why risk parity:
    - Equal weights (10% each) don't mean equal risk contribution
    - A high-volatility factor with 10% weight may contribute 30% of variance
    - Risk parity ensures each factor contributes equally to portfolio risk

    Example:
    - Factor A: 20% vol, Factor B: 10% vol
    - Equal weight: A contributes 4x more risk than B
    - Risk parity: Weight A at ~33%, B at ~67% for equal risk contribution

    Mathematical formulation:
    - Portfolio variance: σ²_p = w'Σw
    - Marginal risk contribution of factor i: MRC_i = (Σw)_i / σ_p
    - Total risk contribution: TRC_i = w_i × MRC_i
    - Risk parity: TRC_i = TRC_j for all i, j
    """

    # Constraints
    MIN_WEIGHT = 0.02  # No factor below 2%
    MAX_WEIGHT = 0.40  # No factor above 40%

    def __init__(
        self,
        factor_returns: Dict[str, List[float]],
        lookback_periods: int = 60,
    ):
        """
        Initialize risk parity weighter.

        Args:
            factor_returns: Dict of factor_name -> list of returns
            lookback_periods: Number of periods for covariance estimation
        """
        self.factor_returns = factor_returns
        self.lookback_periods = lookback_periods

        self.factor_names = list(factor_returns.keys())
        self.returns_matrix = self._build_returns_matrix()
        self.cov_matrix = self._calculate_covariance()

    def _build_returns_matrix(self) -> np.ndarray:
        """Build matrix of factor returns (periods x factors)."""
        if not self.factor_names:
            return np.array([])

        min_len = min(len(r) for r in self.factor_returns.values())
        n_periods = min(min_len, self.lookback_periods)

        matrix = np.zeros((n_periods, len(self.factor_names)))

        for j, factor_name in enumerate(self.factor_names):
            returns = self.factor_returns[factor_name][-n_periods:]
            matrix[:, j] = returns

        return matrix

    def _calculate_covariance(self) -> np.ndarray:
        """Calculate factor covariance matrix."""
        if self.returns_matrix.size == 0:
            return np.array([])

        return np.cov(self.returns_matrix.T)

    def calculate_weights(self) -> Optional[RiskParityWeights]:
        """
        Calculate risk parity weights.

        Returns:
            RiskParityWeights or None if optimization fails
        """
        n_factors = len(self.factor_names)

        if n_factors < 2:
            logger.warning("Need at least 2 factors for risk parity")
            return None

        if self.cov_matrix.size == 0:
            logger.warning("Insufficient return data for covariance")
            return None

        # Ensure covariance matrix is positive semi-definite
        min_eig = np.min(np.linalg.eigvalsh(self.cov_matrix))
        if min_eig < 0:
            self.cov_matrix += (-min_eig + 1e-6) * np.eye(n_factors)

        # Calculate volatilities
        vols = np.sqrt(np.diag(self.cov_matrix))

        # Optimization: minimize sum of squared differences in risk contributions
        def objective(weights):
            """Objective: minimize risk contribution disparity."""
            weights = np.array(weights)
            port_var = weights @ self.cov_matrix @ weights

            if port_var <= 0:
                return 1e10

            port_vol = np.sqrt(port_var)
            marginal_contrib = self.cov_matrix @ weights / port_vol
            risk_contrib = weights * marginal_contrib

            # Target: equal risk contribution
            target_contrib = port_vol / n_factors

            # Sum of squared deviations from equal contribution
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},  # Sum to 1
        ]

        bounds = [(self.MIN_WEIGHT, self.MAX_WEIGHT)] * n_factors

        # Initial guess: inverse volatility weights
        inv_vol_weights = (1.0 / vols) / np.sum(1.0 / vols)

        # Optimize
        result = minimize(
            objective,
            inv_vol_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if not result.success:
            logger.warning(f"Risk parity optimization did not converge: {result.message}")

        weights = result.x

        # Calculate final risk contributions
        port_var = weights @ self.cov_matrix @ weights
        port_vol = np.sqrt(port_var)
        marginal_contrib = self.cov_matrix @ weights / port_vol
        risk_contrib = weights * marginal_contrib

        # Build result
        weight_dict = {
            self.factor_names[i]: float(weights[i])
            for i in range(n_factors)
        }

        risk_dict = {
            self.factor_names[i]: float(risk_contrib[i])
            for i in range(n_factors)
        }

        vol_dict = {
            self.factor_names[i]: float(vols[i])
            for i in range(n_factors)
        }

        rp_result = RiskParityWeights(
            factor_weights=weight_dict,
            risk_contributions=risk_dict,
            factor_volatilities=vol_dict,
            factor_correlations=np.corrcoef(self.returns_matrix.T),
            optimization_converged=result.success,
            timestamp=datetime.now(),
        )

        logger.info(
            f"Risk parity weights: {', '.join(f'{k}={v:.1%}' for k, v in weight_dict.items())}. "
            f"Risk concentration: {rp_result.risk_concentration:.4f}"
        )

        return rp_result


class AdaptiveFactorWeighter:
    """
    Combines orthogonalization and risk parity for optimal factor weighting.

    Process:
    1. Orthogonalize factors to remove correlations
    2. Calculate risk parity weights on orthogonal factors
    3. Map back to original factor weights
    4. Blend with IC-based weights for alpha maximization

    This provides:
    - Diversification benefit (orthogonalization removes redundancy)
    - Risk balance (risk parity equalizes contributions)
    - Alpha focus (IC weighting prioritizes predictive factors)
    """

    # Blending parameter (0 = pure risk parity, 1 = pure IC-based)
    IC_BLEND_WEIGHT = 0.3

    def __init__(
        self,
        factor_scores: Dict[str, Dict[str, float]],
        factor_returns: Dict[str, List[float]],
        ic_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize adaptive weighter.

        Args:
            factor_scores: Current factor scores by symbol
            factor_returns: Historical factor returns
            ic_weights: Optional IC-based weight adjustments
        """
        self.factor_scores = factor_scores
        self.factor_returns = factor_returns
        self.ic_weights = ic_weights or {}

    def calculate_optimal_weights(self) -> Dict[str, float]:
        """
        Calculate optimal factor weights.

        Returns:
            Dict of factor_name -> optimal weight
        """
        # Step 1: Orthogonalize
        orthogonalizer = FactorOrthogonalizer(
            self.factor_scores,
            method=OrthogonalizationMethod.PCA,
        )
        ortho_result = orthogonalizer.orthogonalize()

        # Step 2: Risk parity on orthogonal factors
        if ortho_result and self.factor_returns:
            rp_weighter = RiskParityWeighter(self.factor_returns)
            rp_result = rp_weighter.calculate_weights()

            if rp_result:
                base_weights = rp_result.factor_weights
            else:
                # Fallback to equal weights
                n = len(self.factor_scores)
                base_weights = dict.fromkeys(self.factor_scores.keys(), 1.0 / n)
        else:
            # Fallback to equal weights
            n = len(self.factor_scores)
            base_weights = dict.fromkeys(self.factor_scores.keys(), 1.0 / n)

        # Step 3: Blend with IC weights
        if self.ic_weights:
            final_weights = {}

            for factor in base_weights.keys():
                rp_weight = base_weights.get(factor, 0)
                ic_mult = self.ic_weights.get(factor, 1.0)

                # Blend: (1 - blend) * risk_parity + blend * ic_adjusted
                ic_adjusted = rp_weight * ic_mult
                blended = (1 - self.IC_BLEND_WEIGHT) * rp_weight + self.IC_BLEND_WEIGHT * ic_adjusted
                final_weights[factor] = blended

            # Renormalize
            total = sum(final_weights.values())
            if total > 0:
                final_weights = {k: v / total for k, v in final_weights.items()}

            return final_weights
        else:
            return base_weights

    def get_diagnostic_report(self) -> Dict[str, Any]:
        """
        Generate diagnostic report on factor weighting.

        Returns:
            Dict with orthogonalization and risk parity diagnostics
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "n_factors": len(self.factor_scores),
            "factors": list(self.factor_scores.keys()),
        }

        # Orthogonalization diagnostics
        orthogonalizer = FactorOrthogonalizer(self.factor_scores)
        ortho_result = orthogonalizer.orthogonalize()

        if ortho_result:
            report["orthogonalization"] = {
                "method": ortho_result.method.value,
                "correlation_reduction": ortho_result.correlation_reduction,
                "explained_variance_ratio": ortho_result.explained_variance_ratio[:3],
                "correlations_before": ortho_result.factor_correlations_before.tolist(),
                "correlations_after": ortho_result.factor_correlations_after.tolist(),
            }

        # Risk parity diagnostics
        if self.factor_returns:
            rp_weighter = RiskParityWeighter(self.factor_returns)
            rp_result = rp_weighter.calculate_weights()

            if rp_result:
                report["risk_parity"] = {
                    "weights": rp_result.factor_weights,
                    "risk_contributions": rp_result.risk_contributions,
                    "volatilities": rp_result.factor_volatilities,
                    "risk_concentration": rp_result.risk_concentration,
                    "optimization_converged": rp_result.optimization_converged,
                }

        # Final weights
        report["optimal_weights"] = self.calculate_optimal_weights()

        return report
