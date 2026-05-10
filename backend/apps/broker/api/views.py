from rest_framework import serializers, viewsets

from apps.broker.models import BrokerLink


class BrokerLinkSerializer(serializers.ModelSerializer):
    class Meta:
        model = BrokerLink
        fields = "__all__"


class BrokerLinkViewSet(viewsets.ModelViewSet):
    serializer_class = BrokerLinkSerializer

    def get_queryset(self):
        return BrokerLink.objects.filter(tenant=self.request.tenant)
