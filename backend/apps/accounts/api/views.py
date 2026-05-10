from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.accounts.models import User
from apps.accounts.api.serializers import SignupSerializer, UserSerializer


class AuthViewSet(viewsets.ViewSet):
    permission_classes = [permissions.AllowAny]

    @action(detail=False, methods=["post"])
    def signup(self, request):
        ser = SignupSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        user = ser.save()
        return Response(UserSerializer(user).data, status=status.HTTP_201_CREATED)


class MeViewSet(viewsets.ViewSet):
    def list(self, request):
        return Response(UserSerializer(request.user).data)
