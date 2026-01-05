"""
Callbacks de distillation teacher->student (response-based).

Principe : on ajoute une récompense auxiliaire proportionnelle au log-prob
du teacher sur l'action réalisée, pour encourager la politique student
à imiter les sorties du teacher (logits/probas).
"""
from typing import Optional

import torch as th
from stable_baselines3.common.callbacks import BaseCallback  # type: ignore


class DistillationRewardCallback(BaseCallback):
    """
    Ajoute une récompense de distillation basée sur les sorties du teacher.
    Reward += coef * log_prob_teacher(action | obs)
    """

    def __init__(self, teacher_model, coef: float = 0.1, verbose: int = 0):
        super().__init__(verbose)
        self.teacher = teacher_model
        self.coef = coef

    def _on_rollout_end(self) -> None:  # noqa: D401
        # Utilise le rollout buffer collecté par la politique courante
        try:
            buffer = self.model.rollout_buffer
            # Observations et actions au format torch sur le device du student
            obs = buffer.get_obs()
            actions = buffer.actions
            # Passer le teacher en eval et sur le bon device
            device = self.model.device
            teacher_policy = self.teacher.policy
            teacher_policy.to(device)
            teacher_policy.eval()
            with th.no_grad():
                dist = teacher_policy.get_distribution(obs.to(device))
                logprob = dist.log_prob(actions.to(device))
                # En continu, log_prob retourne [n_steps, n_env, action_dim] ou [n_steps, n_env]
                if logprob.dim() > 2:
                    logprob = logprob.sum(-1)
                logprob = logprob.cpu().numpy()
            buffer.rewards += self.coef * logprob
        except Exception as exc:  # pragma: no cover - sécurité
            if self.verbose > 0:
                print(f"[DistillationRewardCallback] skipped due to: {exc}")

