from django.apps import AppConfig


class CommonConfig(AppConfig):
    name = "apps.common"
    verbose_name = "Common / shared kernel"

    def ready(self) -> None:
        # Install the credential-redaction filter on logzero's logger so the
        # Angel SmartApi SDK can never write API keys / JWTs / login bodies to
        # our logs. logzero is out of reach of Django's LOGGING dictConfig.
        from apps.common.logging import install_credential_redaction

        install_credential_redaction()
