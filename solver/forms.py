from django import forms

class LinearProgrammingForm(forms.Form):
    c1 = forms.FloatField(label="c1 (objective function coefficient for x1)")
    c2 = forms.FloatField(label="c2 (objective function coefficient for x2)")

    # Constraints
    a11 = forms.FloatField(label="a11 (coefficient of x1 in constraint 1)")
    a12 = forms.FloatField(label="a12 (coefficient of x2 in constraint 1)")
    b1 = forms.FloatField(label="b1 (constraint 1 limit)")

    a21 = forms.FloatField(label="a21 (coefficient of x1 in constraint 2)")
    a22 = forms.FloatField(label="a22 (coefficient of x2 in constraint 2)")
    b2 = forms.FloatField(label="b2 (constraint 2 limit)")

    # Additional constraints
    a31 = forms.FloatField(label="a31 (coefficient of x1 in constraint 3)")
    a32 = forms.FloatField(label="a32 (coefficient of x2 in constraint 3)")
    b3 = forms.FloatField(label="b3 (constraint 3 limit)")

    a41 = forms.FloatField(label="a41 (coefficient of x1 in constraint 4)")
    a42 = forms.FloatField(label="a42 (coefficient of x2 in constraint 4)")
    b4 = forms.FloatField(label="b4 (constraint 4 limit)")
