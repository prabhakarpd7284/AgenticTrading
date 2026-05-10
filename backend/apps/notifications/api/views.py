from rest_framework import serializers, viewsets

from apps.notifications.models import Alert


class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = Alert
        fields = "__all__"


class AlertViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = AlertSerializer

    def get_queryset(self):
        return Alert.objects.filter(tenant=self.request.tenant).order_by("-created_at")
