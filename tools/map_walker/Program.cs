using System;
using System.Windows.Forms;

namespace MapWalker;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        ApplicationConfiguration.Initialize();
        using var form = new MainForm();
        Application.Run(form);
    }
}
