from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from win32com.client import gencache
import sys
import matplotlib.pyplot as plt

# Notes
#
# The python project and script was tested with the following tools:
#       Python 3.4.3 for Windows (32-bit) (https://www.python.org/downloads/) - Python interpreter
#       Python for Windows Extensions (32-bit, Python 3.4) (http://sourceforge.net/projects/pywin32/) - for COM support
#       Microsoft Visual Studio Express 2013 for Windows Desktop (https://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx) - easy-to-use IDE
#       Python Tools for Visual Studio (https://pytools.codeplex.com/) - integration into Visual Studio
#
# Note that Visual Studio and Python Tools make development easier, however this python script should should run without either installed.

class PythonStandaloneApplication(object):
    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass

    def __init__(self):
        # make sure the Python wrappers are available for the COM client and
        # interfaces
        gencache.EnsureModule('{EA433010-2BAC-43C4-857C-7AEAC4A8CCE0}', 0, 1, 0)
        gencache.EnsureModule('{F66684D7-AAFE-4A62-9156-FF7A7853F764}', 0, 1, 0)
        # Note - the above can also be accomplished using 'makepy.py' in the
        # following directory:
        #      {PythonEnv}\Lib\site-packages\win32com\client\
        # Also note that the generate wrappers do not get refreshed when the
        # COM library changes.
        # To refresh the wrappers, you can manually delete everything in the
        # cache directory:
        #	   {PythonEnv}\Lib\site-packages\win32com\gen_py\*.*
        
        self.TheConnection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
        if self.TheConnection is None:
            raise PythonStandaloneApplication.ConnectionException("Unable to intialize COM connection to ZOSAPI")

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException("Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication.LicenseException("License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None

        self.TheConnection = None

    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication.InitializationException("Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if self.TheApplication.LicenseStatus is constants.LicenseStatusType_PremiumEdition:
            return "Premium"
        elif self.TheApplication.LicenseStatus is constants.LicenseStatusType_ProfessionalEdition:
            return "Professional"
        elif self.TheApplication.LicenseStatus is constants.LicenseStatusType_StandardEdition:
            return "Standard"
        else:
            return "Invalid"


if __name__ == '__main__':

    zosapi = PythonStandaloneApplication()

    file_name = "\\Users\\L.Desdoigts\\Documents\\Zemax\\Samples\\Sequential\\TolimanZemaxShare\\Toliman-RC-f50_03TryPupilmask.ZMX"
    zosapi.OpenFile(file_name, False)

    # Set up primary optical system
    sys = zosapi.TheSystem
    sys_data = sys.SystemData

    app = zosapi.TheApplication
    # print(dir(sys_data.Fields))

    field_1 = sys_data.Fields.GetField(1)

    # psf = analyses.New_HuygensPsf()

    TheAnalyses = sys.Analyses
    asses = TheAnalyses.New_HuygensPsf()

    settings = asses.GetSettings()
    IAS_settings = CastTo(settings,'IAS_HuygensPsf')

    # print(dir(sys_data.__dict__['_oleobj_'].GetIDsOfNames))
    asses.ApplyAndWaitForCompletion()

    asses_results = asses.GetResults()
    asses_results_cast = CastTo(asses_results, 'IAR_')

    Xs = []
    Ys = []

    print(asses_results_cast.NumberOfDataSeries)

    for seriesNum in range(0,asses_results_cast.NumberOfDataSeries,1):
        data = asses_results_cast.GetDataSeries(seriesNum)
        Xs.append(np.array(data.XData.Data))
        Ys.append(np.array(data.YData.Data))

        # plt.plot(x[:],y[:,0],color=colors[seriesNum])
        # plt.plot(x[:],y[:,1],linestyle='--',color=colors[seriesNum])

    print(Xs)
    print()
    print(Ys)

    # meta_data = asses_results_cast.MetaData
    # print(meta_data)
    # meta_data_cast = CastTo(meta_data, "IAR_")

    # print(dir(meta_data_cast))
    # print()
    # print(meta_data_cast.__dict__)

    # text_file = asses_results_cast.GetTextFile()
    # data_grid = asses_results_cast.DataGrids

    # print(grid_data[0])

    # data_grid = data_grid[0]

    # print(dir(data_grid))


    # data_grid_cast = CastTo(data_grid, "IAR_")

    # print(data_grid_cast)
    # print()
    # print(data_grid_cast.Description)
    # print()
    # print(dir(data_grid_cast))
    # print()
    # print(data_grid_cast.__dict__)






    # analyses = app.PrimarySystem.Analyses
    # psf = analyses.New_HuygensPsf()
    # # psf = analyses.New_FftMtf()
    # psf.Apply()
    # results = psf.GetResults()
    # settings = psf.GetSettings()

    # print(dir(settings))
    # print(settings.ModifySettings())
    # # print(settings.Type)
    # print(dir(psf))
    




    # # print(results)
    # # print(psf.MetaData)

    # print(dir(analyses.New_HuygensPsf().__dict__["_oleobj_"]))
    # print(dir(analyses.New_HuygensPsf().__dict__["_oleobj_"].Invoke()))

    # print(app)
    # obj = app.__dict__["_oleobj_"]
    # print(obj["GetIDsOfNames"])
    # print(dir(app.__dict__["_oleobj_"]))

    # for k, v in zosapi.__dict__.items():
    # 	print(k, v)
    # 	print(v.__dict__)
    # 	print(type(v.__dict__["_oleobj_"]))
    # 	print(dir(v.__dict__["_oleobj_"]))

    # app = zosapi.TheApplication


    # print(zosapi)
    # value = zosapi.ExampleConstants()
    # print(value)
    
    
    # # Analyses = zosapi.Analysis
    
    # SamplesFolder = app.SamplesDir
    # file = "Sequential\\TolimanZemaxShare\\Toliman-RC-f50_03_MirrorSag_MakePSFs_showingLouis.zmx"
    # # sys.path.append("{}//{}".format(SamplesFolder, file))
    
    # primary = app.PrimarySystem
    
    # print(app.LicenseStatus)
    
    # primary.LoadFile("{}//{}".format(SamplesFolder, file))
    
    # # primary_system = zosapi.TheApplication.PrimarySystem
    # # print(ThePrimarySystem.LoadFile(__path_to_file__ )) # <- This should load the file to primary_system as it returns a bool

    # # using zosapi.Analaysis; <- Java version, what is it for python

    # # the_analyses = ThePrimarySystem.Analyses 
        
    
    # # This will clean up the connection to OpticStudio.
    # # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # # this until you need to.
    # del zosapi
    # zosapi = None
    
    print("Done")



