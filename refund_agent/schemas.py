"""Format de la réponse finale de l'agent (sortie structurée Pydantic)."""

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["refund_issued", "not_eligible", "info_only", "need_more_info", "not_found"]


class RefundResponse(BaseModel):
    """Réponse finale de l'agent, dans un format fixe."""

    decision: Decision = Field(
        description="refund_issued: ticket(s) créé(s) ; not_eligible: hors délai ; "
                    "info_only: simple question, aucun ticket ; need_more_info: il "
                    "manque une information ; not_found: client ou article introuvable")
    refund_amount: float | None = Field(
        default=None, description="Montant total remboursé, si des tickets ont été créés")
    ticket_ids: list[str] = Field(default_factory=list, description="Tickets créés")
    message: str = Field(description="Réponse au client, en français")
