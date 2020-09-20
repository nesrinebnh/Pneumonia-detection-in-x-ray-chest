from rest_framework import viewsets
from ..models import Test
from .serializers import TestSerializer

class TestViewSet(viewsets.ModelViewSet):
    serializer_class = TestSerializer
    queryset = Test.objects.all()