from django import forms


class UploadFileForm(forms.Form):
    fileToDeface = forms.FileField()
