# Import domain modules so their @register_domain decorators execute.
from harness.tools.domains import (
    csm,  # noqa: F401
    customer_service,  # noqa: F401
    field_service,  # noqa: F401
    marketing,  # noqa: F401
    sdr,  # noqa: F401
)
