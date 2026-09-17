from rest_framework import serializers

from apps.accounts.models import User


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "email", "full_name", "mfa_enabled", "created_at"]
        read_only_fields = ["id", "created_at", "mfa_enabled"]


class SignupSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=12)

    class Meta:
        model = User
        fields = ["email", "full_name", "password"]

    def create(self, validated):
        user = User.objects.create_user(**validated)
        # Bootstrap the personal tenant+membership right at signup so the
        # *very first* token minted for this user already carries a valid
        # tenant_id claim.  Idempotent.
        from apps.accounts.services.tenant_bootstrap import ensure_tenant
        ensure_tenant(user)
        return user
