import urwid

def main():
    txt = urwid.Text(u"Minimal TUI for testing. Auto-exiting...")
    fill = urwid.Filler(txt, 'top')
    loop = urwid.MainLoop(fill)
    loop.set_alarm_in(0.2, lambda _loop, _data: _loop.stop()) # Auto-exit
    loop.run()

if __name__ == '__main__':
    main()
