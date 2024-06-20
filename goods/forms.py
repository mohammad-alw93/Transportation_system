from django import forms

class ContactForm(forms.Form):
    num_trucks = forms.CharField(required=True)
    truck_capacities = forms.CharField(required=True)
    num_goods = forms.CharField(required=True)
    items_value = forms.CharField(required=True)
    items_weight = forms.CharField(required=True)
    number_addresses = forms.CharField(required=True)
    time_addresses = forms.CharField(required=True)
