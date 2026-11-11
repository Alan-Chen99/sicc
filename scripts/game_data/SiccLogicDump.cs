// BepInEx plugin for the Stationeers dedicated server: once game data is loaded, write the logic
// tables of every ILogicable prefab as JSON to $SICC_DUMP_OUT, then quit.
//
// The tables are computed the way Stationpedia.PopulateThingPages does (AddLogicTypeInfo,
// AddLogicSlotTypeInfo, AddSlotInfo, AddLogicModeInfo). The server builds Stationpedia pages too;
// the computed tables are cross-checked against them, and any mismatch fails the run. The tables
// are computed rather than read off the pages because hidden prefabs have no page.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Assets.Scripts;
using Assets.Scripts.Localization2;
using Assets.Scripts.Objects;
using Assets.Scripts.Objects.Electrical;
using Assets.Scripts.Objects.Motherboards;
using Assets.Scripts.Objects.Pipes;
using Assets.Scripts.UI;
using BepInEx;
using Newtonsoft.Json;
using Objects.Rockets;
using UnityEngine;

[BepInPlugin("sicc.logicdump", "SiccLogicDump", "1.0.0")]
public class SiccLogicDump : BaseUnityPlugin
{
    // Messages go through UnityEngine.Debug (the server's -logFile): BepInEx's own disk log is not
    // reliably flushed before Application.Quit.
    void Awake()
    {
        WorldManager.OnGameDataLoaded += OnGameDataLoaded;
        Debug.Log("SiccLogicDump: waiting for WorldManager.OnGameDataLoaded");
    }

    static void OnGameDataLoaded()
    {
        int code = 0;
        try
        {
            string outPath = Environment.GetEnvironmentVariable("SICC_DUMP_OUT");
            if (string.IsNullOrEmpty(outPath))
                throw new Exception("SICC_DUMP_OUT is not set");
            List<Entry> entries = Collect();
            Write(outPath, entries);
            Debug.Log("SiccLogicDump: wrote " + entries.Count + " prefabs to " + outPath);
            if (!CrossCheck(entries))
                code = 3;
        }
        catch (Exception e)
        {
            Debug.LogError("SiccLogicDump: FAILED\n" + e);
            code = 1;
        }
        Debug.Log("SiccLogicDump: exiting with code " + code);
        Application.Quit(code);
    }

    class Entry
    {
        public Thing Prefab;
        public bool Hidden;
        public List<KeyValuePair<string, string>> Logic = new List<KeyValuePair<string, string>>();
        public List<KeyValuePair<string, List<int>>> SlotLogic = new List<KeyValuePair<string, List<int>>>();
        public List<string> Modes = new List<string>();
    }

    static List<Entry> Collect()
    {
        Stationpedia.DataHandler.HandleThingPageOverrides();
        Debug.Log("SiccLogicDump: prefabs=" + Prefab.AllPrefabs.Count
            + " HiddenInPedia=" + Stationpedia.DataHandler.HiddenInPedia.Count);

        var entries = new List<Entry>();
        foreach (Thing prefab in Prefab.AllPrefabs.OrderBy(p => p.PrefabName, StringComparer.Ordinal))
        {
            ILogicable logicable = prefab as ILogicable;
            if (logicable == null)
                continue;

            bool hiddenOverride;
            Stationpedia.DataHandler.HiddenInPedia.TryGetValue(prefab.PrefabName, out hiddenOverride);
            var e = new Entry { Prefab = prefab, Hidden = prefab.HideInStationpedia || hiddenOverride };

            DeviceInputOutput dio = logicable as DeviceInputOutput;
            if (dio != null)
            {
                foreach (Connection openEnd in dio.OpenEnds)
                    openEnd.Validate();
            }
            foreach (LogicType lt in EnumCollections.LogicTypes.Values)
            {
                bool r = logicable.CanLogicRead(lt);
                bool w = logicable.CanLogicWrite(lt);
                if (r || w)
                    e.Logic.Add(new KeyValuePair<string, string>(lt.ToString(), (r ? "R" : "") + (w ? "W" : "")));
            }

            if (prefab.Slots != null)
            {
                foreach (LogicSlotType st in Logicable.LogicSlotTypes)
                {
                    var idx = new List<int>();
                    for (int i = 0; i < prefab.Slots.Count; i++)
                    {
                        if (prefab.Slots[i] != null && logicable.CanLogicRead(st, i))
                            idx.Add(i);
                    }
                    if (idx.Count > 0)
                        e.SlotLogic.Add(new KeyValuePair<string, List<int>>(st.ToString(), idx));
                }
            }

            if ((prefab is RocketAvionicsDevice || prefab.HasModeState) && !(prefab is LogicDial) && prefab.ModeStrings != null)
                e.Modes.AddRange(prefab.ModeStrings);

            entries.Add(e);
        }
        return entries;
    }

    static void Write(string outPath, List<Entry> entries)
    {
        using (var sw = new StreamWriter(outPath))
        using (var w = new JsonTextWriter(sw))
        {
            w.Formatting = Formatting.Indented;
            w.Indentation = 1;
            w.WriteStartObject();
            w.WritePropertyName("game_version");
            w.WriteValue(GameManager.GetGameVersion());

            w.WritePropertyName("enums");
            w.WriteStartObject();
            WriteEnum<LogicType>(w, "LogicType");
            WriteEnum<LogicSlotType>(w, "LogicSlotType");
            WriteEnum<ColorType>(w, "ColorType");
            w.WriteEndObject();

            w.WritePropertyName("prefabs");
            w.WriteStartArray();
            foreach (Entry e in entries)
            {
                w.WriteStartObject();
                w.WritePropertyName("prefab_name");
                w.WriteValue(e.Prefab.PrefabName);
                w.WritePropertyName("prefab_hash");
                w.WriteValue(e.Prefab.PrefabHash);
                w.WritePropertyName("class");
                w.WriteValue(e.Prefab.GetType().FullName);
                w.WritePropertyName("hidden");
                w.WriteValue(e.Hidden);
                // CableNetwork.DataDeviceList (what batch instructions and pins address) is List<Device>
                w.WritePropertyName("device");
                w.WriteValue(e.Prefab is Device);

                w.WritePropertyName("logic");
                w.WriteStartObject();
                foreach (var kv in e.Logic)
                {
                    w.WritePropertyName(kv.Key);
                    w.WriteValue(kv.Value);
                }
                w.WriteEndObject();

                w.WritePropertyName("slot_logic");
                w.WriteStartObject();
                foreach (var kv in e.SlotLogic)
                {
                    w.WritePropertyName(kv.Key);
                    w.WriteRawValue("[" + string.Join(", ", kv.Value) + "]");
                }
                w.WriteEndObject();

                w.WritePropertyName("slots");
                w.WriteStartArray();
                if (e.Prefab.HasAnySlots)
                {
                    foreach (Slot slot in e.Prefab.Slots)
                        w.WriteRawValue(JsonConvert.SerializeObject(new { name = slot.StringKey, type = slot.Type.ToString() }));
                }
                w.WriteEndArray();

                w.WritePropertyName("modes");
                w.WriteRawValue(JsonConvert.SerializeObject(e.Modes));

                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteEndObject();
            sw.WriteLine();
        }
    }

    static void WriteEnum<T>(JsonTextWriter w, string name) where T : struct
    {
        w.WritePropertyName(name);
        w.WriteStartObject();
        foreach (T v in (T[])Enum.GetValues(typeof(T)))
        {
            w.WritePropertyName(v.ToString());
            w.WriteValue(Convert.ToInt64(v));
        }
        w.WriteEndObject();
    }

    static readonly Regex LogicTypeLink = new Regex("link=LogicType(\\w+)>");
    static readonly Regex LogicSlotTypeLink = new Regex("link=LogicSlotType(\\w+)>");

    static string Access(string pediaAccess)
    {
        if (pediaAccess == GameStrings.LogicRead.DisplayString) return "R";
        if (pediaAccess == GameStrings.LogicWrite.DisplayString) return "W";
        if (pediaAccess == GameStrings.LogicReadWrite.DisplayString) return "RW";
        throw new Exception("unknown Stationpedia access string: " + pediaAccess);
    }

    static string Name(Regex re, string pediaName)
    {
        Match m = re.Match(pediaName);
        if (!m.Success)
            throw new Exception("cannot parse Stationpedia logic name: " + pediaName);
        return m.Groups[1].Value;
    }

    // Returns false on any mismatch between the computed tables and the game's own Stationpedia.
    static bool CrossCheck(List<Entry> entries)
    {
        var pages = new Dictionary<string, StationpediaPage>();
        foreach (StationpediaPage p in Stationpedia.StationpediaPages)
            pages[p.Key] = p;

        int compared = 0, missing = 0, mismatched = 0;
        foreach (Entry e in entries)
        {
            if (e.Hidden)
                continue;
            StationpediaPage page;
            if (!pages.TryGetValue("Thing" + e.Prefab.PrefabName, out page))
            {
                missing++;
                continue;
            }
            compared++;
            string mine = string.Join(" ", e.Logic.Select(kv => kv.Key + ":" + kv.Value))
                + " | " + string.Join(" ", e.SlotLogic.Select(kv => kv.Key + ":" + string.Join(", ", kv.Value)));
            string theirs = string.Join(" ", page.LogicInsert.Select(i => Name(LogicTypeLink, i.LogicName) + ":" + Access(i.LogicAccessTypes)))
                + " | " + string.Join(" ", page.LogicSlotInsert.Select(i => Name(LogicSlotTypeLink, i.LogicName) + ":" + i.LogicAccessTypes));
            if (mine != theirs)
            {
                mismatched++;
                if (mismatched <= 20)
                    Debug.LogError("SiccLogicDump: MISMATCH " + e.Prefab.PrefabName + "\n  computed:    " + mine + "\n  stationpedia: " + theirs);
            }
        }
        Debug.Log("SiccLogicDump: cross-check: Stationpedia.Instance=" + (Stationpedia.Instance != null)
            + " pages=" + pages.Count + " compared=" + compared + " no_page=" + missing + " mismatched=" + mismatched);
        if (compared == 0)
            Debug.LogError("SiccLogicDump: no Stationpedia pages to cross-check against");
        return compared > 0 && mismatched == 0;
    }
}
