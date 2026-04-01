# Token-level case studies

Source directory: `/Users/michelpicker/Library/Mobile Documents/com~apple~CloudDocs/Projekte/thesis_MPhil/github/results/2026-03-22_verified/explainability`

## global_art_tfidf_sgd / high_confidence_correct

- SHA256: `e1a94beaaf6f39dfabf4ea7eed5c47b59009e0be426156e8d0d986f763c42cb5`
- Date: `2019-11-11 00:00:00`
- True family: `emotet`
- Predicted family: `emotet`
- Correct: `True`
- Confidence: `0.984122`
- Margin: `0.980306`

Supporting tokens:
- `artifact::exefile` (art_tfidf, contribution=1.1206)
- `artifact::systemfileassociations` (art_tfidf, contribution=0.9411)
- `artifact::identifier` (art_tfidf, contribution=0.5334)
- `artifact::global` (art_tfidf, contribution=0.4447)
- `artifact::hkey_classes_root` (art_tfidf, contribution=0.3965)
- `artifact::mutex` (art_tfidf, contribution=0.3243)

Opposing tokens:
- `artifact::hkey_current_user` (art_tfidf, contribution=-0.3631)
- `artifact::microsoft` (art_tfidf, contribution=-0.2841)
- `artifact::software` (art_tfidf, contribution=-0.1838)
- `artifact::windows` (art_tfidf, contribution=-0.1608)
- `artifact::tmp` (art_tfidf, contribution=-0.1456)
- `artifact::dll` (art_tfidf, contribution=-0.1389)

## global_art_tfidf_sgd / low_confidence_correct

- SHA256: `e505ed734d2149c6596f2314840e37825555e99990e42e205607caa73d42e557`
- Date: `2019-09-17 00:00:00`
- True family: `swisyn`
- Predicted family: `swisyn`
- Correct: `True`
- Confidence: `0.184674`
- Margin: `0.034753`

Supporting tokens:
- `artifact::hlp` (art_tfidf, contribution=0.1418)
- `artifact::ini` (art_tfidf, contribution=0.0786)
- `artifact::icsys` (art_tfidf, contribution=0.0635)
- `artifact::icn` (art_tfidf, contribution=0.0635)
- `artifact::xa0` (art_tfidf, contribution=0.0627)
- `artifact::desktop` (art_tfidf, contribution=0.0560)

Opposing tokens:
- `artifact::file` (art_tfidf, contribution=-0.0740)
- `artifact::dll` (art_tfidf, contribution=-0.0670)
- `artifact::txt` (art_tfidf, contribution=-0.0556)
- `artifact::file_ext` (art_tfidf, contribution=-0.0503)
- `artifact::dat` (art_tfidf, contribution=-0.0289)
- `artifact::exe` (art_tfidf, contribution=-0.0176)

## global_art_tfidf_sgd / high_confidence_error

- SHA256: `4a4a0b451a08883c491b083d0f6a281f09a4b184aa7b9f24647fb63d0c10f697`
- Date: `2019-09-23 00:00:00`
- True family: `trickbot`
- Predicted family: `njrat`
- Correct: `False`
- Confidence: `0.906693`
- Margin: `0.88629`

Supporting tokens:
- `artifact::netframework` (art_tfidf, contribution=1.0378)
- `artifact::mscorrc` (art_tfidf, contribution=0.8770)
- `artifact::pi` (art_tfidf, contribution=0.4522)
- `artifact::config` (art_tfidf, contribution=0.3455)
- `artifact::mscoreei` (art_tfidf, contribution=0.3041)
- `artifact::__tmp_rar_sfx_access_check_` (art_tfidf, contribution=0.2931)

Opposing tokens:
- `artifact::file` (art_tfidf, contribution=-0.2002)
- `artifact::hkey_local_machine` (art_tfidf, contribution=-0.1681)
- `artifact::software` (art_tfidf, contribution=-0.1595)
- `artifact::microsoft` (art_tfidf, contribution=-0.1564)
- `artifact::nt` (art_tfidf, contribution=-0.1347)
- `artifact::reg` (art_tfidf, contribution=-0.1340)

## global_art_tfidf_sgd / borderline_case

- SHA256: `270995c92bd208edd784555d54c8d508487a9e7fdd48cd6f053956ce3f74a07e`
- Date: `2019-10-25 00:00:00`
- True family: `swisyn`
- Predicted family: `ursnif`
- Correct: `False`
- Confidence: `0.159856`
- Margin: `0.002272`

Supporting tokens:
- `artifact::bmp` (art_tfidf, contribution=0.0419)
- `artifact::cat` (art_tfidf, contribution=0.0351)
- `artifact::file` (art_tfidf, contribution=0.0331)
- `artifact::tmp` (art_tfidf, contribution=0.0246)
- `artifact::shdocvw` (art_tfidf, contribution=0.0236)
- `artifact::sdb` (art_tfidf, contribution=0.0117)

Opposing tokens:
- `artifact::exe` (art_tfidf, contribution=-0.0796)
- `artifact::file_ext` (art_tfidf, contribution=-0.0779)
- `artifact::inf` (art_tfidf, contribution=-0.0675)
- `artifact::hlp` (art_tfidf, contribution=-0.0568)
- `artifact::sys` (art_tfidf, contribution=-0.0448)
- `artifact::db` (art_tfidf, contribution=-0.0395)

## per_family_api_tfidf_sgd / high_confidence_correct

- SHA256: `8ac99b4dff975d9173b50d20d6000661c97bfa14678691fa11d25869f4ab426d`
- Date: `2019-11-08 00:00:00`
- True family: `emotet`
- Predicted family: `emotet`
- Correct: `True`
- Confidence: `0.988585`
- Margin: `0.986068`

Supporting tokens:
- `api::propsys` (api_tfidf, contribution=0.3251)
- `api::propsys dll` (api_tfidf, contribution=0.3251)
- `api::virtualalloc` (api_tfidf, contribution=0.0761)
- `api::dll virtualalloc` (api_tfidf, contribution=0.0761)
- `api::regqueryvalueexa kernel32` (api_tfidf, contribution=0.0690)
- `api::gettemppatha kernel32` (api_tfidf, contribution=0.0658)

Opposing tokens:
- `api::advapi32 dll` (api_tfidf, contribution=-0.0720)
- `api::advapi32` (api_tfidf, contribution=-0.0720)
- `api::cryptsp dll` (api_tfidf, contribution=-0.0521)
- `api::cryptsp` (api_tfidf, contribution=-0.0521)
- `api::sechost dll` (api_tfidf, contribution=-0.0306)
- `api::sechost` (api_tfidf, contribution=-0.0306)

## per_family_api_tfidf_sgd / low_confidence_correct

- SHA256: `0a2e32465434580b82bae447ac648d825080d5934c5ff4d1e25721e1963c08a4`
- Date: `2019-10-26 00:00:00`
- True family: `zeus`
- Predicted family: `zeus`
- Correct: `True`
- Confidence: `0.160929`
- Margin: `0.006975`

Supporting tokens:
- none

Opposing tokens:
- none

## per_family_api_tfidf_sgd / high_confidence_error

- SHA256: `ed256ea8995a49ef927041a1d976f3a27e28adcce8b809d80e24b5ab0391fefa`
- Date: `2019-09-29 00:00:00`
- True family: `njrat`
- Predicted family: `swisyn`
- Correct: `False`
- Confidence: `0.862194`
- Margin: `0.810916`

Supporting tokens:
- `api::corevokeinitializespy comctl32` (api_tfidf, contribution=0.0386)
- `api::388 oleaut32` (api_tfidf, contribution=0.0382)
- `api::dll nlsgetcacheupdatecount` (api_tfidf, contribution=0.0333)
- `api::nlsgetcacheupdatecount` (api_tfidf, contribution=0.0333)
- `api::386 ole32` (api_tfidf, contribution=0.0309)
- `api::500 advapi32` (api_tfidf, contribution=0.0300)

Opposing tokens:
- `api::dll` (api_tfidf, contribution=-0.3615)
- `api::kernel32` (api_tfidf, contribution=-0.1307)
- `api::kernel32 dll` (api_tfidf, contribution=-0.1307)
- `api::getmonitorinfoa kernel32` (api_tfidf, contribution=-0.1000)
- `api::shellexecutea setupapi` (api_tfidf, contribution=-0.0606)
- `api::user32 dll` (api_tfidf, contribution=-0.0450)

## per_family_api_tfidf_sgd / borderline_case

- SHA256: `32f75d4a63798dc132ad36560c7cd6447f363f1a4347359dc19f8e71f02aab7f`
- Date: `2019-11-29 00:00:00`
- True family: `trickbot`
- Predicted family: `lokibot`
- Correct: `False`
- Confidence: `0.253401`
- Margin: `0.001312`

Supporting tokens:
- `api::crypthashdata cryptsp` (api_tfidf, contribution=0.0608)
- `api::cryptdestroyhash cryptsp` (api_tfidf, contribution=0.0581)
- `api::imm32` (api_tfidf, contribution=0.0523)
- `api::imm32 dll` (api_tfidf, contribution=0.0523)
- `api::dll crypthashdata` (api_tfidf, contribution=0.0453)
- `api::crypthashdata` (api_tfidf, contribution=0.0453)

Opposing tokens:
- `api::dll` (api_tfidf, contribution=-0.1372)
- `api::kernel32 dll` (api_tfidf, contribution=-0.0860)
- `api::kernel32` (api_tfidf, contribution=-0.0860)
- `api::rpcrt4 dll` (api_tfidf, contribution=-0.0210)
- `api::rpcrt4` (api_tfidf, contribution=-0.0210)
- `api::dll cryptencrypt` (api_tfidf, contribution=-0.0196)

## per_family_art_tfidf_sgd / high_confidence_correct

- SHA256: `e1a94beaaf6f39dfabf4ea7eed5c47b59009e0be426156e8d0d986f763c42cb5`
- Date: `2019-11-11 00:00:00`
- True family: `emotet`
- Predicted family: `emotet`
- Correct: `True`
- Confidence: `0.985818`
- Margin: `0.98241`

Supporting tokens:
- `artifact::exefile` (art_tfidf, contribution=1.2021)
- `artifact::systemfileassociations` (art_tfidf, contribution=1.0045)
- `artifact::identifier` (art_tfidf, contribution=0.5642)
- `artifact::global` (art_tfidf, contribution=0.4256)
- `artifact::hkey_classes_root` (art_tfidf, contribution=0.3787)
- `artifact::mutex` (art_tfidf, contribution=0.3255)

Opposing tokens:
- `artifact::hkey_current_user` (art_tfidf, contribution=-0.3682)
- `artifact::microsoft` (art_tfidf, contribution=-0.3002)
- `artifact::software` (art_tfidf, contribution=-0.2203)
- `artifact::windows` (art_tfidf, contribution=-0.1751)
- `artifact::file` (art_tfidf, contribution=-0.1589)
- `artifact::tmp` (art_tfidf, contribution=-0.1476)

## per_family_art_tfidf_sgd / low_confidence_correct

- SHA256: `82011dae0f88539a743704874e3573d3a18f9b86ba0fde5d03eb55aa8abb8db8`
- Date: `2019-11-17 00:00:00`
- True family: `swisyn`
- Predicted family: `swisyn`
- Correct: `True`
- Confidence: `0.174522`
- Margin: `0.006129`

Supporting tokens:
- `artifact::hlp` (art_tfidf, contribution=0.1407)
- `artifact::ini` (art_tfidf, contribution=0.1065)
- `artifact::desktop` (art_tfidf, contribution=0.0819)
- `artifact::icsys` (art_tfidf, contribution=0.0676)
- `artifact::icn` (art_tfidf, contribution=0.0676)
- `artifact::xa0` (art_tfidf, contribution=0.0667)

Opposing tokens:
- `artifact::mui` (art_tfidf, contribution=-0.0771)
- `artifact::file` (art_tfidf, contribution=-0.0771)
- `artifact::dll` (art_tfidf, contribution=-0.0681)
- `artifact::txt` (art_tfidf, contribution=-0.0627)
- `artifact::file_ext` (art_tfidf, contribution=-0.0498)
- `artifact::propsys` (art_tfidf, contribution=-0.0435)

## per_family_art_tfidf_sgd / high_confidence_error

- SHA256: `8c4812ecae08c29702d22ca1617b3d2a9198d77fcd6229d5e473ffd61a7d19df`
- Date: `2019-10-24 00:00:00`
- True family: `emotet`
- Predicted family: `trickbot`
- Correct: `False`
- Confidence: `0.895607`
- Margin: `0.872527`

Supporting tokens:
- `artifact::sqmclient` (art_tfidf, contribution=2.5272)
- `artifact::nt` (art_tfidf, contribution=0.5670)
- `artifact::appid` (art_tfidf, contribution=0.4001)
- `artifact::software` (art_tfidf, contribution=0.2713)
- `artifact::microsoft` (art_tfidf, contribution=0.2598)
- `artifact::rpc` (art_tfidf, contribution=0.2068)

Opposing tokens:
- `artifact::com3` (art_tfidf, contribution=-0.2763)
- `artifact::hkey_current_user` (art_tfidf, contribution=-0.2708)
- `artifact::file` (art_tfidf, contribution=-0.2505)
- `artifact::reg` (art_tfidf, contribution=-0.1499)
- `artifact::hkey_classes_root` (art_tfidf, contribution=-0.1489)
- `artifact::file_ext` (art_tfidf, contribution=-0.1142)

## per_family_art_tfidf_sgd / borderline_case

- SHA256: `f97156216ef9051fa4d199bc719434923f1645fa4a5f0738d36711bcdf105103`
- Date: `2019-05-02 00:00:00`
- True family: `zeus`
- Predicted family: `zeus`
- Correct: `True`
- Confidence: `0.211032`
- Margin: `0.00029`

Supporting tokens:
- `artifact::tmp` (art_tfidf, contribution=0.2580)
- `artifact::services` (art_tfidf, contribution=0.1613)
- `artifact::nt` (art_tfidf, contribution=0.1242)
- `artifact::file` (art_tfidf, contribution=0.1232)
- `artifact::mutex` (art_tfidf, contribution=0.1124)
- `artifact::microsoft` (art_tfidf, contribution=0.0724)

Opposing tokens:
- `artifact::dat` (art_tfidf, contribution=-0.3188)
- `artifact::nls` (art_tfidf, contribution=-0.2940)
- `artifact::hkey_current_user` (art_tfidf, contribution=-0.1485)
- `artifact::sortdefault` (art_tfidf, contribution=-0.1463)
- `artifact::file_ext` (art_tfidf, contribution=-0.1366)
- `artifact::mpr` (art_tfidf, contribution=-0.1192)

## table_5_11_fusion_global_sgd / high_confidence_correct

- SHA256: `ad73cb1c533e99d22d30e36949aa1ebecd5be118315b8a713e6ae414ce3e1a0c`
- Date: `2019-11-12 00:00:00`
- True family: `emotet`
- Predicted family: `emotet`
- Correct: `True`
- Confidence: `0.992338`
- Margin: `0.990833`

Supporting tokens:
- `artifact::exefile` (art_tfidf, contribution=0.5826)
- `artifact::systemfileassociations` (art_tfidf, contribution=0.5181)
- `artifact::classes` (art_tfidf, contribution=0.2967)
- `artifact::hkey_classes_root` (art_tfidf, contribution=0.2887)
- `artifact::identifier` (art_tfidf, contribution=0.2703)
- `artifact::global` (art_tfidf, contribution=0.2554)

Opposing tokens:
- `artifact::hkey_current_user` (art_tfidf, contribution=-0.1935)
- `artifact::microsoft` (art_tfidf, contribution=-0.1721)
- `artifact::dll` (art_tfidf, contribution=-0.1201)
- `artifact::file` (art_tfidf, contribution=-0.1144)
- `artifact::windows` (art_tfidf, contribution=-0.0936)
- `artifact::reg` (art_tfidf, contribution=-0.0854)

## table_5_11_fusion_global_sgd / low_confidence_correct

- SHA256: `a9b59bdf1395f7275793421158548494d92646662939550080fe325673fa81f3`
- Date: `2019-09-27 00:00:00`
- True family: `emotet`
- Predicted family: `emotet`
- Correct: `True`
- Confidence: `0.170738`
- Margin: `0.00928`

Supporting tokens:
- `artifact::nls` (art_tfidf, contribution=0.0880)
- `artifact::control` (art_tfidf, contribution=0.0708)
- `artifact::controlset001` (art_tfidf, contribution=0.0541)
- `artifact::sortdefault` (art_tfidf, contribution=0.0535)
- `artifact::system` (art_tfidf, contribution=0.0372)
- `api::isprocessdpiaware dwmapi` (api_tfidf, contribution=0.0328)

Opposing tokens:
- `artifact::file` (art_tfidf, contribution=-0.0850)
- `artifact::microsoft` (art_tfidf, contribution=-0.0660)
- `artifact::file_ext` (art_tfidf, contribution=-0.0646)
- `artifact::reg` (art_tfidf, contribution=-0.0437)
- `artifact::tmp` (art_tfidf, contribution=-0.0334)
- `api::dll` (api_tfidf, contribution=-0.0326)

## table_5_11_fusion_global_sgd / high_confidence_error

- SHA256: `3d8bc4aeb9ad4317f3e8bae4bcff3ef0f9ed3ddd7d79dbcadc4ff1c0e4842596`
- Date: `2019-10-26 00:00:00`
- True family: `harhar`
- Predicted family: `njrat`
- Correct: `False`
- Confidence: `0.862426`
- Margin: `0.803486`

Supporting tokens:
- `artifact::cdf` (art_tfidf, contribution=0.2259)
- `artifact::vbs` (art_tfidf, contribution=0.1923)
- `artifact::bat` (art_tfidf, contribution=0.1686)
- `artifact::pi` (art_tfidf, contribution=0.1245)
- `artifact::_system32_` (art_tfidf, contribution=0.1056)
- `artifact::startup` (art_tfidf, contribution=0.0884)

Opposing tokens:
- `artifact::set` (art_tfidf, contribution=-0.1004)
- `artifact::file` (art_tfidf, contribution=-0.0879)
- `artifact::nt` (art_tfidf, contribution=-0.0722)
- `artifact::hkey_local_machine` (art_tfidf, contribution=-0.0709)
- `artifact::reg` (art_tfidf, contribution=-0.0679)
- `artifact::db` (art_tfidf, contribution=-0.0634)

## table_5_11_fusion_global_sgd / borderline_case

- SHA256: `f1e2e1a9f542954c017e627cedb9ccde92ffe466e7bca9b37ac18f5d41abc495`
- Date: `2019-10-10 00:00:00`
- True family: `emotet`
- Predicted family: `lokibot`
- Correct: `False`
- Confidence: `0.267165`
- Margin: `0.00038`

Supporting tokens:
- `artifact::dat` (art_tfidf, contribution=0.0694)
- `artifact::staticcache` (art_tfidf, contribution=0.0656)
- `api::gdi32 dll` (api_tfidf, contribution=0.0634)
- `api::gdi32` (api_tfidf, contribution=0.0634)
- `artifact::oleaccrc` (art_tfidf, contribution=0.0422)
- `api::imm32 dll` (api_tfidf, contribution=0.0376)

Opposing tokens:
- `artifact::reg` (art_tfidf, contribution=-0.0990)
- `artifact::mutex` (art_tfidf, contribution=-0.0814)
- `artifact::global` (art_tfidf, contribution=-0.0681)
- `api::dll` (api_tfidf, contribution=-0.0631)
- `artifact::exe` (art_tfidf, contribution=-0.0598)
- `api::kernel32 dll` (api_tfidf, contribution=-0.0511)

## table_5_12_fusion_per_family_sgd / high_confidence_correct

- SHA256: `ad73cb1c533e99d22d30e36949aa1ebecd5be118315b8a713e6ae414ce3e1a0c`
- Date: `2019-11-12 00:00:00`
- True family: `emotet`
- Predicted family: `emotet`
- Correct: `True`
- Confidence: `0.993815`
- Margin: `0.992699`

Supporting tokens:
- `artifact::exefile` (art_tfidf, contribution=0.6280)
- `artifact::systemfileassociations` (art_tfidf, contribution=0.5622)
- `artifact::identifier` (art_tfidf, contribution=0.2861)
- `artifact::hkey_classes_root` (art_tfidf, contribution=0.2662)
- `artifact::global` (art_tfidf, contribution=0.2495)
- `artifact::classes` (art_tfidf, contribution=0.2299)

Opposing tokens:
- `artifact::hkey_current_user` (art_tfidf, contribution=-0.2038)
- `artifact::microsoft` (art_tfidf, contribution=-0.1699)
- `artifact::file` (art_tfidf, contribution=-0.1117)
- `artifact::dll` (art_tfidf, contribution=-0.1022)
- `artifact::software` (art_tfidf, contribution=-0.0935)
- `artifact::windows` (art_tfidf, contribution=-0.0906)

## table_5_12_fusion_per_family_sgd / low_confidence_correct

- SHA256: `088759cfbe7232091e249d40c702b9cc8bde4421c5529fb509514351b9af005f`
- Date: `2019-11-06 00:00:00`
- True family: `emotet`
- Predicted family: `emotet`
- Correct: `True`
- Confidence: `0.205916`
- Margin: `0.027674`

Supporting tokens:
- `api::imm32 dll` (api_tfidf, contribution=0.0968)
- `api::imm32` (api_tfidf, contribution=0.0968)
- `artifact::wow6432node` (art_tfidf, contribution=0.0366)
- `api::dll processidtosessionid` (api_tfidf, contribution=0.0229)
- `api::processidtosessionid` (api_tfidf, contribution=0.0229)
- `api::initializecriticalsectionandspincount kernel32` (api_tfidf, contribution=0.0201)

Opposing tokens:
- `artifact::dll` (art_tfidf, contribution=-0.1557)
- `artifact::microsoft` (art_tfidf, contribution=-0.1301)
- `artifact::file` (art_tfidf, contribution=-0.0978)
- `artifact::hkey_current_user` (art_tfidf, contribution=-0.0828)
- `artifact::tmp` (art_tfidf, contribution=-0.0644)
- `artifact::software` (art_tfidf, contribution=-0.0639)

## table_5_12_fusion_per_family_sgd / high_confidence_error

- SHA256: `ff122bf4096e27d7490d98ffa49795735938294ed5955852ff2928001a3e9cc8`
- Date: `2019-07-14 00:00:00`
- True family: `lokibot`
- Predicted family: `njrat`
- Correct: `False`
- Confidence: `0.934591`
- Margin: `0.904026`

Supporting tokens:
- `artifact::eventvwr` (art_tfidf, contribution=0.4525)
- `artifact::cdf` (art_tfidf, contribution=0.2997)
- `artifact::sdb` (art_tfidf, contribution=0.2433)
- `artifact::mouse` (art_tfidf, contribution=0.2195)
- `artifact::mscfile` (art_tfidf, contribution=0.2123)
- `artifact::autoit` (art_tfidf, contribution=0.2116)

Opposing tokens:
- `artifact::hkey_local_machine` (art_tfidf, contribution=-0.1081)
- `artifact::file` (art_tfidf, contribution=-0.0887)
- `artifact::db` (art_tfidf, contribution=-0.0860)
- `artifact::software` (art_tfidf, contribution=-0.0834)
- `artifact::classes` (art_tfidf, contribution=-0.0711)
- `artifact::microsoft` (art_tfidf, contribution=-0.0683)

## table_5_12_fusion_per_family_sgd / borderline_case

- SHA256: `9e423109191fab5cd1557b8772402d862125f5cd9e0034bee7e74d87d3e308ee`
- Date: `2019-10-25 00:00:00`
- True family: `emotet`
- Predicted family: `emotet`
- Correct: `True`
- Confidence: `0.37381`
- Margin: `0.0`

Supporting tokens:
- `artifact::msptermtangent` (art_tfidf, contribution=0.4120)
- `artifact::global` (art_tfidf, contribution=0.2435)
- `artifact::mutex` (art_tfidf, contribution=0.1852)
- `artifact::classes` (art_tfidf, contribution=0.1848)
- `artifact::hkey_classes_root` (art_tfidf, contribution=0.1234)
- `artifact::cmd` (art_tfidf, contribution=0.1063)

Opposing tokens:
- `artifact::sqmclient` (art_tfidf, contribution=-0.3842)
- `artifact::dat` (art_tfidf, contribution=-0.2205)
- `artifact::microsoft` (art_tfidf, contribution=-0.1627)
- `artifact::hkey_current_user` (art_tfidf, contribution=-0.1531)
- `artifact::ole` (art_tfidf, contribution=-0.1523)
- `artifact::rpc` (art_tfidf, contribution=-0.1171)

## table_5_8 / high_confidence_correct

- SHA256: `d322879374b6d4d57ae4a1d99db4b69ccf77adffa1ffd6c2fc587a065ebfed74`
- Date: `2019-10-06 00:00:00`
- True family: `emotet`
- Predicted family: `emotet`
- Correct: `True`
- Confidence: `0.990096`
- Margin: `0.987956`

Supporting tokens:
- `api::propsys` (api_tfidf, contribution=0.3281)
- `api::propsys dll` (api_tfidf, contribution=0.3281)
- `api::wsprintfa kernel32` (api_tfidf, contribution=0.0979)
- `api::wsprintfa` (api_tfidf, contribution=0.0813)
- `api::dll wsprintfa` (api_tfidf, contribution=0.0813)
- `api::freeconsole user32` (api_tfidf, contribution=0.0790)

Opposing tokens:
- `api::advapi32 dll` (api_tfidf, contribution=-0.0769)
- `api::advapi32` (api_tfidf, contribution=-0.0769)
- `api::cryptsp dll` (api_tfidf, contribution=-0.0508)
- `api::cryptsp` (api_tfidf, contribution=-0.0508)
- `api::sechost` (api_tfidf, contribution=-0.0307)
- `api::sechost dll` (api_tfidf, contribution=-0.0307)

## table_5_8 / low_confidence_correct

- SHA256: `0a2e32465434580b82bae447ac648d825080d5934c5ff4d1e25721e1963c08a4`
- Date: `2019-10-26 00:00:00`
- True family: `zeus`
- Predicted family: `zeus`
- Correct: `True`
- Confidence: `0.197672`
- Margin: `0.055757`

Supporting tokens:
- none

Opposing tokens:
- none

## table_5_8 / high_confidence_error

- SHA256: `ed256ea8995a49ef927041a1d976f3a27e28adcce8b809d80e24b5ab0391fefa`
- Date: `2019-09-29 00:00:00`
- True family: `njrat`
- Predicted family: `swisyn`
- Correct: `False`
- Confidence: `0.843953`
- Margin: `0.795066`

Supporting tokens:
- `api::corevokeinitializespy comctl32` (api_tfidf, contribution=0.0397)
- `api::388 oleaut32` (api_tfidf, contribution=0.0385)
- `api::dll nlsgetcacheupdatecount` (api_tfidf, contribution=0.0370)
- `api::nlsgetcacheupdatecount` (api_tfidf, contribution=0.0370)
- `api::386 ole32` (api_tfidf, contribution=0.0306)
- `api::500 advapi32` (api_tfidf, contribution=0.0299)

Opposing tokens:
- `api::dll` (api_tfidf, contribution=-0.3365)
- `api::kernel32` (api_tfidf, contribution=-0.1229)
- `api::kernel32 dll` (api_tfidf, contribution=-0.1229)
- `api::getmonitorinfoa kernel32` (api_tfidf, contribution=-0.0915)
- `api::shellexecutea setupapi` (api_tfidf, contribution=-0.0544)
- `api::user32 dll` (api_tfidf, contribution=-0.0441)

## table_5_8 / borderline_case

- SHA256: `c1f3e58f1a5be8b856800fdd12d85195661f7fde5258c417b9a7a472ff1c37d0`
- Date: `2019-10-26 00:00:00`
- True family: `zeus`
- Predicted family: `njrat`
- Correct: `False`
- Confidence: `0.179976`
- Margin: `7.2e-05`

Supporting tokens:
- `api::getfileattributesexw kernel32` (api_tfidf, contribution=0.0140)
- `api::lcmapstringa kernel32` (api_tfidf, contribution=0.0135)
- `api::dll lcmapstringa` (api_tfidf, contribution=0.0135)
- `api::lcmapstringa` (api_tfidf, contribution=0.0135)
- `api::localfiletimetofiletime kernel32` (api_tfidf, contribution=0.0132)
- `api::localfiletimetofiletime` (api_tfidf, contribution=0.0132)

Opposing tokens:
- `api::dll` (api_tfidf, contribution=-0.0647)
- `api::kernel32` (api_tfidf, contribution=-0.0630)
- `api::kernel32 dll` (api_tfidf, contribution=-0.0630)
- `api::virtualprotect kernel32` (api_tfidf, contribution=-0.0100)
- `api::virtualprotect` (api_tfidf, contribution=-0.0088)
- `api::dll virtualprotect` (api_tfidf, contribution=-0.0088)

