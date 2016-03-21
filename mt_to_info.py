#! /usr/bin/env python
import sys
import pyfits as fits


def main():
    """Gets the info from fits file and writes it to a file."""
    if len(sys.argv[1:]) > 0:
        for dwfile in sys.argv[1:]:
            print dwfile
            dwinfo = dwfile.split("/")[-1]
            if ".fits" in dwfile:
                dwinfo = dwinfo.replace(".fits", ".info")
            else:
                dwinfo += ".info"

            vtowrtie = []
            vvars = []
            vtowrtie.append("\nVariables:\n")
            d = fits.getdata(dwfile, 1)
            for name, minval, maxval, num in zip(d['NAME'], d['MINIMUM'], d['MAXIMUM'], d['NUMBVALS']):
                vvars.append(name.lower())
                vtowrtie.append("%s: %.3f : %.3f  %i\n" % (name, minval, maxval, num))

            with open(dwinfo, "w") as fw:
                dwheader = fits.getheader(dwfile, 0)
                fw.write("%s: %s /%s\n" % ('MODLNAME', dwheader['MODLNAME'], "first 12 characters of model name"))
                dwheader = fits.getheader(dwfile, 1)
                fw.write("\nInput vals(if also in Variables then ignore this value):\n")
                for para, pinfo in [('GAMMA', 'photon index'), ('LOGT', 'log T_e not useful anymore as it\'s calculated'), ('D', 'D parameter'), ('R_IN', 'R_in/rg'), ('R_OUT', 'R_out/R_in'), ('VINF', 'Vinf/Vesc'), ('VSCALER', 'VscaleR/R_out'), ('CLUMPING', 'Clumping parameter'), ('Fe-grp abund', '[Fe/H]'), ('NINTPARM', 'number of interpolated parameters'), ('NADDPARM', 'number of additional parameters'), ('ELOW', 'energy band low end (eV)'), ('EHIGH', 'energy band high end (eV)'), ('M_BH', 'black hole mass/solar'), ('LUM', 'L_x/L_Edd'), ('OBOUND', 'outer boundary/rg'), ('REM', 'size of emission region/rg'), ('V0', 'launch velocity'), ('VBETA', 'velocity law beta'), ('KVAL', 'mass-loss exponent')]:
                    if para.lower() not in vvars:
                        fw.write("%s: %s /%s\n" % (para, str(dwheader[para]), pinfo))

                fw.writelines(vtowrtie)

if __name__ == '__main__':
    main()
