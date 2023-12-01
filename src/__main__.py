import PySimpleGUI as sg

layout = [
    [sg.Text("I exist as a text")],
    [sg.Button("Push me if you feel like it")],
          ]

window = sg.Window("I am Groot?", layout=layout, margins=(500,300))

while True:
    event, values = window.read()
    if event == "Push me if you feel like it" or event == sg.WINDOW_CLOSED:
        break
