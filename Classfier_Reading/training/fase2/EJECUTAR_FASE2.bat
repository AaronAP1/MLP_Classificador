@echo off
chcp 65001 >nul
echo ================================================================================
echo 🚀 EJECUTOR FASE 2 - ENTRENAMIENTO COMPLETO
echo ================================================================================
echo.
echo Este script ejecutará el entrenamiento de los 9 modelos (3 grupos × 3 arquitecturas)
echo usando SOLAMENTE variables demográficas.
echo.
echo Tiempo estimado: 3-4 horas
echo.
pause

cd /d "%~dp0"
python run_all_fase2.py

echo.
echo ================================================================================
echo ✅ Ejecución finalizada
echo ================================================================================
echo.
echo Revisa los resultados en: outputs\fase2\
echo.
pause
