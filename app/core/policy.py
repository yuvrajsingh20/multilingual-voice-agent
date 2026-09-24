from app.core.state import ConversationState


def evaluate_policy(state: ConversationState) -> dict:
    return {
        "allow_collection": True,
        "require_confirmation": False,
        "escalate": state.escalation_required,
    }
