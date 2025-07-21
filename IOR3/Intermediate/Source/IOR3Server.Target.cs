using UnrealBuildTool;

public class IOR3ServerTarget : TargetRules
{
	public IOR3ServerTarget(TargetInfo Target) : base(Target)
	{
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		Type = TargetType.Server;
		ExtraModuleNames.Add("IOR3");
	}
}
