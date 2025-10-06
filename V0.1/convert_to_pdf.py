#!/usr/bin/env python3
"""
Script para convertir el informe HTML a PDF
Requiere instalar: pip install weasyprint
Alternativa: pip install pdfkit wkhtmltopdf
"""

import os
import sys
from pathlib import Path

def convert_html_to_pdf_weasyprint():
    """Convierte HTML a PDF usando WeasyPrint"""
    try:
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        html_file = "informe_rendimiento_modelos.html"
        pdf_file = "informe_rendimiento_modelos.pdf"
        
        if not os.path.exists(html_file):
            print(f"❌ Error: No se encontró el archivo {html_file}")
            return False
            
        print(f"🔄 Convirtiendo {html_file} a PDF...")
        
        # CSS adicional para PDF
        pdf_css = CSS(string='''
            @page {
                size: A4;
                margin: 2cm;
            }
            
            .container {
                max-width: none;
                margin: 0;
            }
            
            .model-card {
                break-inside: avoid;
                page-break-inside: avoid;
            }
            
            .section {
                break-inside: avoid;
            }
            
            .ranking-table {
                break-inside: avoid;
            }
            
            h2 {
                page-break-after: avoid;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .conclusions {
                page-break-before: auto;
            }
        ''')
        
        # Configuración de fuentes
        font_config = FontConfiguration()
        
        # Convertir HTML a PDF
        html_doc = HTML(filename=html_file)
        html_doc.write_pdf(
            pdf_file, 
            stylesheets=[pdf_css],
            font_config=font_config
        )
        
        print(f"✅ PDF generado exitosamente: {pdf_file}")
        return True
        
    except ImportError:
        print("❌ WeasyPrint no está instalado.")
        print("📦 Instalar con: pip install weasyprint")
        return False
    except Exception as e:
        print(f"❌ Error al convertir con WeasyPrint: {e}")
        return False

def convert_html_to_pdf_pdfkit():
    """Convierte HTML a PDF usando pdfkit (requiere wkhtmltopdf)"""
    try:
        import pdfkit
        
        html_file = "informe_rendimiento_modelos.html"
        pdf_file = "informe_rendimiento_modelos.pdf"
        
        if not os.path.exists(html_file):
            print(f"❌ Error: No se encontró el archivo {html_file}")
            return False
            
        print(f"🔄 Convirtiendo {html_file} a PDF con pdfkit...")
        
        # Opciones para wkhtmltopdf
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None
        }
        
        # Convertir HTML a PDF
        pdfkit.from_file(html_file, pdf_file, options=options)
        
        print(f"✅ PDF generado exitosamente: {pdf_file}")
        return True
        
    except ImportError:
        print("❌ pdfkit no está instalado.")
        print("📦 Instalar con: pip install pdfkit")
        print("🔧 También requiere wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
        return False
    except Exception as e:
        print(f"❌ Error al convertir con pdfkit: {e}")
        return False

def convert_html_to_pdf_playwright():
    """Convierte HTML a PDF usando Playwright"""
    try:
        from playwright.sync_api import sync_playwright
        
        html_file = "informe_rendimiento_modelos.html"
        pdf_file = "informe_rendimiento_modelos.pdf"
        
        if not os.path.exists(html_file):
            print(f"❌ Error: No se encontró el archivo {html_file}")
            return False
            
        print(f"🔄 Convirtiendo {html_file} a PDF con Playwright...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # Cargar HTML
            html_path = Path(html_file).absolute().as_uri()
            page.goto(html_path)
            
            # Generar PDF
            page.pdf(
                path=pdf_file,
                format='A4',
                margin={
                    'top': '2cm',
                    'bottom': '2cm',
                    'left': '2cm',
                    'right': '2cm'
                },
                print_background=True
            )
            
            browser.close()
        
        print(f"✅ PDF generado exitosamente: {pdf_file}")
        return True
        
    except ImportError:
        print("❌ Playwright no está instalado.")
        print("📦 Instalar con: pip install playwright")
        print("🔧 Después ejecutar: playwright install chromium")
        return False
    except Exception as e:
        print(f"❌ Error al convertir con Playwright: {e}")
        return False

def main():
    """Función principal - intenta diferentes métodos de conversión"""
    print("📊 GENERADOR DE PDF - INFORME DE RENDIMIENTO")
    print("=" * 50)
    
    # Cambiar al directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Intentar diferentes métodos en orden de preferencia
    methods = [
        ("WeasyPrint", convert_html_to_pdf_weasyprint),
        ("Playwright", convert_html_to_pdf_playwright),
        ("pdfkit", convert_html_to_pdf_pdfkit)
    ]
    
    for method_name, method_func in methods:
        print(f"\n🔧 Intentando con {method_name}...")
        if method_func():
            print(f"\n🎉 ¡Conversión exitosa con {method_name}!")
            return
    
    print("\n❌ No se pudo convertir a PDF con ningún método.")
    print("\n💡 Soluciones alternativas:")
    print("1. Abrir el archivo HTML en tu navegador")
    print("2. Usar Ctrl+P (Imprimir) y 'Guardar como PDF'")
    print("3. Instalar alguna de las librerías mencionadas")

if __name__ == "__main__":
    main()