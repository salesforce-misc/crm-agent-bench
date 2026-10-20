# Import domain modules so their @register_domain decorators execute.
from harness.tools.domains import (
    canonical,  # noqa: F401
    commerce,  # noqa: F401
    csm,  # noqa: F401
    customer_service,  # noqa: F401
    field_service,  # noqa: F401
    integration,  # noqa: F401
    itsm,  # noqa: F401
    marketing,  # noqa: F401
    onboarding,  # noqa: F401
    payments,  # noqa: F401
    sdr,  # noqa: F401
    support,  # noqa: F401
    travel,  # noqa: F401
)
