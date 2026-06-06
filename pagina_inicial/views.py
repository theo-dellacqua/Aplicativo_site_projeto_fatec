from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from .forms import LoginUsuarioForm, CadastroUsuarioForm

def home(request):
    return render(request, 'pagina_inicial/home.html')

def login_usuario(request):
    if request.method == "GET":
        return render(request, "pagina_inicial/login_usuario.html", { "form": LoginUsuarioForm, "next": request.GET.get("next") })
    else:
        post = request.POST
        form = LoginUsuarioForm(post)
        username = post.get("username")
        password = post.get("password")
        next = post.get("next")
        user = authenticate(username=username, password=password)
        if form.is_valid() and user:
            login(request, user)
            if next != "None":
                return redirect(next)
            else:
                return redirect("/")
        else:
            form.add_error("password", "")
            form.add_error("username", "")
            for k in form.errors.keys():
                form.fields[k].widget.attrs['class'] += ' is-invalid'
            form.add_error("", "Usuário / Senha inválido")
            return render(request, "pagina_inicial/login_usuario.html", { "form": form, "next":next })
 

def logout_usuario(request):
    logout(request)
    return redirect("/")


def cadastro_usuario(request):
    if request.method == "GET":
        return render(request, "pagina_inicial/cadastro_usuario.html", { "form": CadastroUsuarioForm })
    else:
        post = request.POST
        form = CadastroUsuarioForm(post)
        if form.is_valid():
            user = User.objects.create_user(
                username = post.get("username"),
                email = post.get("email"),
                first_name = post.get("first_name"),
                last_name = post.get("last_name"),
                password = post.get("password")
            )
            user = authenticate(
                username=post.get("username"),
                password=post.get("password")
            )
            login(request, user)
            return redirect("/")
        else:
            for k in form.errors.keys():
                form.fields[k].widget.attrs['class'] += ' is-invalid'
            form.add_error(None, "Erro(s) foram encontrado(s) ao criar a conta. Favor fazer a devidas correções e tentar novamente.")
            return render(request, "pagina_inicial/cadastro_usuario.html", { "form": form })
        