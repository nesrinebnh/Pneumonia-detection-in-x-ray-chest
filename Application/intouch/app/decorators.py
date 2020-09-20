from django.http import HttpResponse
from django.shortcuts import redirect

def unauthenticated_user(view_func):
	def wrapper_func(request, *args, **kwargs):
		if request.user.is_authenticated:
			if request.user.groups.exists():
				group = request.user.groups.all()[0].name
			if group == 'doctor':
				return redirect('home')

			if group == 'patient':
				return redirect('patient_home')
		else:
			return view_func(request, *args, **kwargs)

	return wrapper_func


def allowed_users(allowed_roles=[]):
	def decorator(view_func):
		def wrapper_func(request, *args, **kwargs):

			group = None
			if request.user.groups.exists():
				group = request.user.groups.all()[0].name

			if group in allowed_roles:
				return view_func(request, *args, **kwargs)
			else:
				return HttpResponse('You are not authorized to view this page')
		return wrapper_func
	return decorator

def patient_only(view_func):
	def wrapper_function(request, *args, **kwargs):
		group = None
		if request.user.groups.exists():
			group = request.user.groups.all()[0].name

		if group == 'doctor':
			return redirect('home')

		if group == 'patient':
			return view_func(request, *args, **kwargs)

	return wrapper_function