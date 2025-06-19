"""
Causal inference utility functions.
"""
from ..utils.pymc import build_and_sample_model

import pandas as pd
import numpy as np

import pymc as pm

import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx

class MediationAnalysis:
    """
    A class for performing Bayesian mediation analysis using a joint mediation model
    that is agnostic to having a single or multiple mediators.
    
    For a single mediator the model is:
      Mediator:    M = α_m + a * X + error
      Outcome:     Y = α_y + c′ * X + b * M + error
      
    For multiple mediators the joint model is:
      For each mediator i:
        M_i = α_m[i] + a[i] * X + error_i
      Outcome:
        Y = α_y + c′ * X + sum_i (b[i] * M_i) + error
      Derived parameters:
        - Individual indirect effects: ab[i] = a[i] * b[i]
        - Total Indirect effect: total_ab = sum_i (a[i]*b[i])
        - Total effect: c = c′ + total_ab
        
    Additionally, if multiple mediators are provided, the class will also fit separate individual 
    models for each mediator.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the predictor, mediator(s), and outcome variables.
    x : str
        Column name for the predictor (X).
    m : str or list of str
        Column name(s) for the mediator(s). If a string is passed, it is converted into a list.
    y : str
        Column name for the outcome (Y).
    hdi : float, optional
        Credible interval width for HDI (default 0.95).
    sampler_kwargs : dict, optional
        Additional keyword arguments for the sampler.
        Default: {"tune": 1000, "draws": 500, "chains": 4,
                  "random_seed": 42, "target_accept": 0.9,
                  "nuts_sampler": "numpyro", "progressbar": False}
    """
    def __init__(self, data: pd.DataFrame, x: str, m, y: str, hdi: float = 0.95, sampler_kwargs: dict = None):
        self.data = data
        self.x = x
        # If m is a string, convert to list for unified processing.
        self.m_list = [m] if isinstance(m, str) else m
        self.y = y
        self.hdi = hdi
        self.sampler_kwargs = sampler_kwargs or {
            "tune": 1000,
            "draws": 500,
            "chains": 4,
            "random_seed": 42,
            "target_accept": 0.9,
            "nuts_sampler": "numpyro",
            "progressbar": False
        }
        # Will store models and inference data for both individual and joint analyses.
        self.models = {"individual": {}, "joint": None}
        self.idata = {"individual": {}, "joint": None}

    def build_model(self):
        """
        Build the mediation models.
        
        For each mediator in self.m_list, an individual model is built.
        A joint model including all mediators is also built.
        """
        # ---------------------
        # Build individual models for each mediator
        # ---------------------
        for mediator in self.m_list:
            X_data = self.data[self.x].values
            M_data = self.data[mediator].values
            Y_data = self.data[self.y].values
            N = len(X_data)
            
            with pm.Model() as model:
                # Mediator equation: M = α_m + a * X + error
                alpha_m = pm.Normal("alpha_m", mu=0.0, sigma=1.0)
                a = pm.Normal("a", mu=0.0, sigma=1.0)
                sigma_m = pm.Exponential("sigma_m", lam=1.0)
                mu_m = alpha_m + a * X_data
                pm.Normal("M_obs", mu=mu_m, sigma=sigma_m, observed=M_data)
    
                # Outcome equation: Y = α_y + c′ * X + b * M + error
                alpha_y = pm.Normal("alpha_y", mu=0.0, sigma=1.0)
                c_prime = pm.Normal("c_prime", mu=0.0, sigma=1.0)
                b = pm.Normal("b", mu=0.0, sigma=1.0)
                sigma_y = pm.Exponential("sigma_y", lam=1.0)
                mu_y = alpha_y + c_prime * X_data + b * M_data
                pm.Normal("Y_obs", mu=mu_y, sigma=sigma_y, observed=Y_data)
    
                # Derived parameters: indirect effect and total effect
                pm.Deterministic("ab", a * b)
                pm.Deterministic("c", c_prime + a * b)
                
            self.models["individual"][mediator] = model

        # ---------------------
        # Build joint model for multiple mediators (or single mediator as joint model)
        # ---------------------
        X_data = self.data[self.x].values
        Y_data = self.data[self.y].values
        # Extract mediators as a matrix (shape: N x K)
        M_data = self.data[self.m_list].values  
        N = len(X_data)
        K = len(self.m_list)
        
        with pm.Model() as joint_model:
            # Vectorized mediator equations: for each mediator i,
            # M_i = α_m[i] + a[i] * X + error
            alpha_m = pm.Normal("alpha_m", mu=0.0, sigma=1.0, shape=K)
            a = pm.Normal("a", mu=0.0, sigma=1.0, shape=K)
            sigma_m = pm.Exponential("sigma_m", lam=1.0, shape=K)
            # Broadcasting X_data to shape (N, 1)
            mu_m = alpha_m + a * X_data[:, None]
            pm.Normal("M_obs", mu=mu_m, sigma=sigma_m, observed=M_data)
    
            # Outcome equation:
            # Y = α_y + c′ * X + sum_i (b[i] * M_i) + error
            alpha_y = pm.Normal("alpha_y", mu=0.0, sigma=1.0)
            c_prime = pm.Normal("c_prime", mu=0.0, sigma=1.0)
            b = pm.Normal("b", mu=0.0, sigma=1.0, shape=K)
            sigma_y = pm.Exponential("sigma_y", lam=1.0)
            mu_y = alpha_y + c_prime * X_data + pm.math.dot(M_data, b)
            pm.Normal("Y_obs", mu=mu_y, sigma=sigma_y, observed=Y_data)
    
            # Derived parameters:
            # Individual indirect effects for each mediator:
            ab = a * b
            pm.Deterministic("ab", ab)
            # Total indirect effect:
            total_ab = pm.math.sum(ab)
            pm.Deterministic("total_ab", total_ab)
            # Total effect:
            pm.Deterministic("c", c_prime + total_ab)
            
        self.models["joint"] = joint_model

    def fit(self):
        """
        Sample from all the built models (individual and joint).
        
        Raises
        ------
        ValueError
            If the models have not been built yet.
        """
        if not self.models["individual"] or self.models["joint"] is None:
            raise ValueError("Models have not been built. Call build_model() before fit().")
        
        # Fit individual models
        for mediator, model in self.models["individual"].items():
            with model:
                self.idata["individual"][mediator] = pm.sample(**self.sampler_kwargs)
                
        # Fit joint model
        with self.models["joint"]:
            self.idata["joint"] = pm.sample(**self.sampler_kwargs)

    def get_summary_individual(self, mediator: str):
        """
        Get numerical summary for the individual model for a given mediator.
        
        Returns
        -------
        dict
            Dictionary with mean estimates and HDI bounds for each parameter.
        """
        var_names = ["alpha_m", "a", "alpha_y", "c_prime", "b", "ab", "c"]
        summary_df = az.summary(self.idata["individual"][mediator], var_names=var_names, hdi_prob=self.hdi)
    
        lower_percent = (1 - self.hdi) / 2 * 100
        upper_percent = 100 - lower_percent
        lower_col = f"hdi_{lower_percent:.1f}%"
        upper_col = f"hdi_{upper_percent:.1f}%"
    
        results = {}
        for key in var_names:
            results[key] = {
                "mean": summary_df.loc[key, "mean"],
                "hdi_lower": summary_df.loc[key, lower_col],
                "hdi_upper": summary_df.loc[key, upper_col]
            }
        return results

    def get_summary_joint(self):
        """
        Get numerical summary for the joint model.
        
        Returns
        -------
        dict
            Dictionary with summary statistics for all parameters.
        """
        summary_df = az.summary(self.idata["joint"], hdi_prob=self.hdi)
    
        lower_percent = (1 - self.hdi) / 2 * 100
        upper_percent = 100 - lower_percent
        lower_col = f"hdi_{lower_percent:.1f}%"
        upper_col = f"hdi_{upper_percent:.1f}%"
    
        results = {}
        for param in summary_df.index:
            results[param] = {
                "mean": summary_df.loc[param, "mean"],
                "hdi_lower": summary_df.loc[param, lower_col],
                "hdi_upper": summary_df.loc[param, upper_col]
            }
        return results

    def get_report_individual(self, mediator: str, x_label: str = None, m_label: str = None, y_label: str = None):
        """
        Generate a plain-language report for an individual mediation analysis.
        
        Parameters
        ----------
        mediator : str
            Mediator column name.
        x_label, m_label, y_label : str, optional
            Labels for the predictor, mediator, and outcome variables.
            
        Returns
        -------
        str
            A human-readable summary of the mediation effects.
        """
        x_label = x_label or self.x
        m_label = m_label or mediator
        y_label = y_label or self.y
        summary = self.get_summary_individual(mediator)
    
        # Extract summary statistics for key parameters
        a_mean = summary["a"]["mean"]
        b_mean = summary["b"]["mean"]
        c_prime_mean = summary["c_prime"]["mean"]
        ab_mean = summary["ab"]["mean"]
        c_mean = summary["c"]["mean"]
    
        # Helper function to check if HDI includes zero
        def hdi_includes_zero(param_stats):
            return param_stats["hdi_lower"] <= 0 <= param_stats["hdi_upper"]
    
        a_zero = hdi_includes_zero(summary["a"])
        b_zero = hdi_includes_zero(summary["b"])
        c_prime_zero = hdi_includes_zero(summary["c_prime"])
        ab_zero = hdi_includes_zero(summary["ab"])
        c_zero = hdi_includes_zero(summary["c"])
    
        lines = []
        lines.append(f"**Bayesian Mediation Analysis for mediator '{m_label}'** ({int(self.hdi * 100)}% HDI)")
        lines.append(f"Variables: {x_label} (predictor), {m_label} (mediator), {y_label} (outcome).")
    
        # Interpret each path
        if not a_zero:
            direction = "positive" if a_mean > 0 else "negative"
            lines.append(f"- Path a ({x_label} → {m_label}) is credibly {direction} (mean = {a_mean:.3f}).")
        else:
            lines.append(f"- Path a ({x_label} → {m_label}) is weak (HDI includes 0, mean = {a_mean:.3f}).")
    
        if not b_zero:
            direction = "positive" if b_mean > 0 else "negative"
            lines.append(f"- Path b ({m_label} → {y_label}, controlling for {x_label}) is credibly {direction} (mean = {b_mean:.3f}).")
        else:
            lines.append(f"- Path b ({m_label} → {y_label}, controlling for {x_label}) is weak (HDI includes 0, mean = {b_mean:.3f}).")
    
        if not ab_zero:
            direction = "positive" if ab_mean > 0 else "negative"
            lines.append(f"- Indirect effect (a×b) is credibly {direction} (mean = {ab_mean:.3f}).")
        else:
            lines.append(f"- Indirect effect (a×b) is uncertain (HDI includes 0, mean = {ab_mean:.3f}).")
    
        if not c_prime_zero:
            direction = "positive" if c_prime_mean > 0 else "negative"
            lines.append(f"- Direct effect (c') is credibly {direction} (mean = {c_prime_mean:.3f}).")
        else:
            lines.append(f"- Direct effect (c') is near zero (HDI includes 0, mean = {c_prime_mean:.3f}).")
    
        if not c_zero:
            direction = "positive" if c_mean > 0 else "negative"
            lines.append(f"- Total effect (c) is credibly {direction} (mean = {c_mean:.3f}).")
        else:
            lines.append(f"- Total effect (c) is uncertain (HDI includes 0, mean = {c_mean:.3f}).")
    
        lines.append("")
        if not ab_zero and c_prime_zero:
            lines.append(f"It appears that {m_label} fully mediates the effect of {x_label} on {y_label}.")
        elif not ab_zero and not c_prime_zero:
            lines.append(f"It appears that {m_label} partially mediates the effect of {x_label} on {y_label}.")
        else:
            lines.append("Mediation is unclear or absent (the indirect effect includes zero or the total effect is not clearly different from zero).")
    
        return "\n".join(lines)
    
    def get_report_joint(self, x_label: str = None, m_label: str = None, y_label: str = None):
        """
        Generate a plain-language report for the joint mediation analysis.
        
        Parameters
        ----------
        x_label, m_label, y_label : str, optional
            Labels for the predictor, mediators (a comma‐separated list), and outcome variables.
            
        Returns
        -------
        str
            A human-readable summary of the joint mediation effects.
        """
        x_label = x_label or self.x
        # If no mediator label is provided, join the mediator names with commas.
        m_label = m_label or ", ".join(self.m_list)
        y_label = y_label or self.y
        summary = self.get_summary_joint()
    
        # For the joint model, we focus on the aggregated effects.
        c_prime_mean = summary["c_prime"]["mean"]
        total_ab_mean = summary["total_ab"]["mean"]
        c_mean = summary["c"]["mean"]
    
        c_prime_zero = summary["c_prime"]["hdi_lower"] <= 0 <= summary["c_prime"]["hdi_upper"]
        total_ab_zero = summary["total_ab"]["hdi_lower"] <= 0 <= summary["total_ab"]["hdi_upper"]
        c_zero = summary["c"]["hdi_lower"] <= 0 <= summary["c"]["hdi_upper"]
    
        lines = []
        lines.append(f"**Joint Bayesian Mediation Analysis** ({int(self.hdi * 100)}% HDI)")
        lines.append(f"Variables: {x_label} (predictor), {m_label} (mediators), {y_label} (outcome).")
    
        if not total_ab_zero:
            direction = "positive" if total_ab_mean > 0 else "negative"
            lines.append(f"- Total Indirect effect (sum of a×b) is credibly {direction} (mean = {total_ab_mean:.3f}).")
        else:
            lines.append(f"- Total Indirect effect (sum of a×b) is uncertain (HDI includes 0, mean = {total_ab_mean:.3f}).")
    
        if not c_prime_zero:
            direction = "positive" if c_prime_mean > 0 else "negative"
            lines.append(f"- Direct effect (c') is credibly {direction} (mean = {c_prime_mean:.3f}).")
        else:
            lines.append(f"- Direct effect (c') is near zero (HDI includes 0, mean = {c_prime_mean:.3f}).")
    
        if not c_zero:
            direction = "positive" if c_mean > 0 else "negative"
            lines.append(f"- Total effect (c) is credibly {direction} (mean = {c_mean:.3f}).")
        else:
            lines.append(f"- Total effect (c) is uncertain (HDI includes 0, mean = {c_mean:.3f}).")
    
        lines.append("")
        if not total_ab_zero and c_prime_zero:
            lines.append(f"It appears that the mediators fully mediate the effect of {x_label} on {y_label}.")
        elif not total_ab_zero and not c_prime_zero:
            lines.append(f"It appears that the mediators partially mediate the effect of {x_label} on {y_label}.")
        else:
            lines.append("Joint mediation is unclear or absent (the total indirect effect includes zero or the total effect is not clearly different from zero).")
    
        return "\n".join(lines)
    
    def print_full_report(self, x_label: str = None, y_label: str = None):
        """
        Print a full report including individual and joint mediation analyses.
        """
        reports = []
        # Individual mediator reports
        for mediator in self.m_list:
            reports.append(self.get_report_individual(mediator, x_label=x_label, y_label=y_label))
        # Joint model report
        reports.append(self.get_report_joint(x_label=x_label, y_label=y_label))
    
        full_report = "\n\n".join(reports)
        print(full_report)

    def get_mediation_type(self,):
        """
        Determine the type of mediation based on the joint model.
        """
        summary = self.get_summary_joint()
        
        total_ab_zero = summary["total_ab"]["hdi_lower"] <= 0 <= summary["total_ab"]["hdi_upper"]
        c_prime_zero = summary["c_prime"]["hdi_lower"] <= 0 <= summary["c_prime"]["hdi_upper"]
        
        if not total_ab_zero and c_prime_zero:
            return "full"
        elif not total_ab_zero and not c_prime_zero:
            return "partial"
        else:
            return "absent"


class ParentCandidateIdentifier:
    def __init__(self, data: pd.DataFrame, node: str, possible_parents: list, epsilon: float = 0.005):
        """
        Parameters:
            data: DataFrame containing your data.
            node: The target variable for which to identify candidate parents.
            possible_parents: A list of potential parent variable names.
            epsilon: Threshold to define "mass around zero" (default 0.05).
        """
        self.data = data
        self.node = node
        self.possible_parents = possible_parents
        self.epsilon = epsilon
        self.runs = {}
        self.results = None

    def build_and_sample_model(self, formula: str):
        """Wrapper for the sampling function."""
        return build_and_sample_model(self.data, formula)

    def compute_mass_around_zero(self, idata, real_mean):
        """
        Compute the fraction of posterior predictive likelihood samples
        (averaged over dates) within epsilon of the real mean.
        """
        estimated_mean = idata.posterior_predictive.likelihood.mean(dim=["date"]).values.flatten()
        distribution = estimated_mean - real_mean
        mass = np.mean(np.abs(distribution) < self.epsilon)
        return mass, distribution

    def run_all_models(self):
        """
        Run the intercept-only model and each individual parent's model,
        storing the sampling results, mass, and error distributions.
        """
        real_mean = self.data[self.node].mean()
        runs = {}

        # Intercept-only model: P(node)
        formula_intercept = f"{self.node} ~ 1"
        idata_int, _ = self.build_and_sample_model(formula_intercept)
        mass_int, dist_int = self.compute_mass_around_zero(idata_int, real_mean)
        runs["intercept_only"] = {
            "formula": formula_intercept,
            "idata": idata_int,
            "mass": mass_int,
            "distribution": dist_int
        }

        # Individual candidate parent models: P(node|parent)
        for parent in self.possible_parents:
            formula_parent = f"{self.node} ~ {parent} + 1"
            idata_parent, _ = self.build_and_sample_model(formula_parent)
            mass_parent, dist_parent = self.compute_mass_around_zero(idata_parent, real_mean)
            runs[f"parent_{parent}"] = {
                "formula": formula_parent,
                "idata": idata_parent,
                "mass": mass_parent,
                "distribution": dist_parent
            }

        self.runs = runs
        return runs

    def identify_candidate_parents(self):
        """
        Runs all models (if not already run), compares the mass around zero,
        and returns a decision: if the intercept-only model is best, the target
        is independent; otherwise, return the candidate parent with the highest mass.
        """
        if not self.runs:
            self.run_all_models()

        best_key, best_info = max(self.runs.items(), key=lambda x: x[1]["mass"])

        if best_key == "intercept_only":
            decision = "independent"
            candidate_parents = []
        else:
            decision = "dependent"
            candidate_parents = [best_key.split("_", 1)[1]]

        self.results = {
            "results": self.runs,
            "best_model": {best_key: best_info},
            "decision": decision,
            "candidate_parents": candidate_parents
        }
        return self.results

    def plot_distributions(self):
        """
        Plot the error distributions from the stored runs using Seaborn.
        """
        if not self.runs:
            self.run_all_models()

        plt.figure(figsize=(10, 5))
        for key, run in self.runs.items():
            sns.kdeplot(run["distribution"], label=run["formula"], fill=True)
        plt.axvline(0, color='red', linestyle='--', label='Zero Error')
        plt.xlabel("Error (Estimated Mean - Real Mean)")
        plt.ylabel("Density")
        plt.title("Posterior Predictive Error Distributions")
        plt.legend()
        plt.show()

def graphviz_to_networkx(graphviz_graph):
    """
    Convert a Graphviz directed graph to a NetworkX directed graph.
    
    Parameters
    ----------
    graphviz_graph : graphviz.Digraph
        The input Graphviz directed graph
        
    Returns
    -------
    nx.DiGraph
        The equivalent NetworkX directed graph
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node in graphviz_graph.body:
        if node.startswith('\t"'):
            # Extract node name
            node_name = node.split('"')[1]
            G.add_node(node_name)
    
    # Add edges
    for edge in graphviz_graph.body:
        if ' -> ' in edge:
            # Extract source and target nodes
            source, target = edge.split(' -> ')
            source = source.strip('\t"')
            target = target.strip(' "\n')
            G.add_edge(source, target)
            
    return G