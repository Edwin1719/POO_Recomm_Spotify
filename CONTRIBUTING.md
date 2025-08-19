# 🤝 Contributing to Spotify Recommender Pro

¡Gracias por tu interés en contribuir al proyecto! 🎉 Este documento proporciona pautas y instrucciones para contribuir de manera efectiva.

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [Cómo Contribuir](#cómo-contribuir)
- [Reporte de Bugs](#reporte-de-bugs)
- [Solicitud de Features](#solicitud-de-features)
- [Pull Requests](#pull-requests)
- [Estilo de Código](#estilo-de-código)
- [Configuración del Entorno](#configuración-del-entorno)
- [Testing](#testing)

## 📜 Código de Conducta

Este proyecto y todos los participantes están regidos por nuestro [Código de Conducta](CODE_OF_CONDUCT.md). Al participar, se espera que mantengas este código. Por favor reporta comportamientos inaceptables a egqa1975@gmail.com.

## 🚀 Cómo Contribuir

### Tipos de Contribuciones Bienvenidas

#### 🐛 **Reportes de Bugs**
- Errores en el sistema de recomendación
- Problemas de UI/UX
- Issues de performance
- Errores de datos

#### ✨ **Nuevas Features**
- Nuevos algoritmos de recomendación
- Mejoras en la interfaz de usuario
- Nuevas visualizaciones
- Integración con APIs externas

#### 📚 **Documentación**
- Mejoras en README
- Documentación de código
- Tutoriales y ejemplos
- Traducciones

#### 🧹 **Code Quality**
- Refactoring
- Optimización de performance
- Mejores prácticas
- Tests adicionales

## 🐛 Reporte de Bugs

### Antes de Reportar un Bug

1. **Busca issues existentes** para evitar duplicados
2. **Verifica la versión** más reciente del proyecto
3. **Reproduce el error** en un entorno limpio

### Template para Bug Reports

```markdown
## 🐛 Descripción del Bug
Una descripción clara y concisa del problema.

## 🔄 Pasos para Reproducir
1. Ir a '...'
2. Hacer click en '...'
3. Scroll hacia abajo hasta '...'
4. Ver el error

## 🎯 Resultado Esperado
Descripción clara de lo que esperabas que sucediera.

## 📱 Screenshots
Si aplica, agrega screenshots para ayudar a explicar el problema.

## 🖥️ Entorno
- OS: [e.g. Windows 10, macOS 11.2, Ubuntu 20.04]
- Python Version: [e.g. 3.9.7]
- Streamlit Version: [e.g. 1.32.0]
- Browser: [e.g. Chrome 96.0, Firefox 94.0]

## 📋 Información Adicional
Cualquier otro contexto sobre el problema.
```

## ✨ Solicitud de Features

### Template para Feature Requests

```markdown
## 🚀 Feature Request

### 📝 Descripción
Una descripción clara de la funcionalidad que te gustaría ver implementada.

### 💡 Motivación
¿Por qué es útil esta feature? ¿Qué problema resuelve?

### 🎯 Solución Propuesta
Descripción detallada de cómo crees que debería funcionar.

### 🔄 Alternativas Consideradas
Otras soluciones o features que has considerado.

### 📊 Contexto Adicional
Screenshots, mockups, o cualquier otro contexto útil.
```

## 🔄 Pull Requests

### Proceso de Pull Request

1. **Fork** el repositorio
2. **Crea una branch** desde `main`
3. **Implementa** tus cambios
4. **Agrega tests** si es apropiado
5. **Actualiza documentación** si es necesario
6. **Envía el Pull Request**

### Naming Convention para Branches

```bash
# Features
feature/add-neural-embeddings
feature/improve-ui-dashboard

# Bug fixes
fix/similarity-calculation-error
fix/streamlit-caching-issue

# Documentation
docs/update-readme
docs/add-api-documentation

# Refactoring
refactor/optimize-data-processing
refactor/clean-recommendation-engine
```

### Template para Pull Requests

```markdown
## 📋 Descripción
Descripción clara de los cambios realizados.

## 🔗 Issue Relacionado
Fixes #(número del issue)

## 🧪 Tipo de Cambio
- [ ] Bug fix (cambio no-breaking que arregla un issue)
- [ ] Nueva feature (cambio no-breaking que agrega funcionalidad)
- [ ] Breaking change (fix o feature que causaría que funcionalidad existente no funcione como se espera)
- [ ] Cambio de documentación

## ✅ Checklist
- [ ] Mi código sigue las convenciones de estilo del proyecto
- [ ] He realizado un self-review de mi código
- [ ] He comentado mi código, particularmente en áreas difíciles de entender
- [ ] He hecho cambios correspondientes a la documentación
- [ ] Mis cambios no generan nuevos warnings
- [ ] He agregado tests que prueban que mi fix es efectivo o que mi feature funciona
- [ ] Tests unitarios nuevos y existentes pasan localmente con mis cambios

## 🧪 Testing
Descripción de cómo fueron probados los cambios.

## 📱 Screenshots (si aplica)
Screenshots de los cambios en la UI.
```

## 🎨 Estilo de Código

### Python Code Style

Seguimos [PEP 8](https://www.python.org/dev/peps/pep-0008/) con algunas modificaciones:

#### **Formateo**
```python
# Usar Black para formateo automático
black --line-length 88 .

# Verificar estilo con flake8
flake8 --max-line-length 88 --ignore E203,W503 .
```

#### **Naming Conventions**
```python
# Variables y funciones: snake_case
user_recommendation = get_user_tracks()

# Clases: PascalCase
class SpotifyRecommender:
    pass

# Constantes: UPPER_SNAKE_CASE
MAX_RECOMMENDATIONS = 20
DEFAULT_SIMILARITY_THRESHOLD = 0.8

# Archivos: snake_case
recommender_optimized.py
```

#### **Docstrings**
```python
def recommend_similar_tracks(self, track_name: str, n: int = 10) -> pd.DataFrame:
    """
    Recommend similar tracks using cosine similarity.
    
    Args:
        track_name (str): Name of the reference track
        n (int): Number of recommendations to return
        
    Returns:
        pd.DataFrame: DataFrame with recommended tracks and similarity scores
        
    Raises:
        ValueError: If track_name is not found in dataset
        
    Example:
        >>> recommender.recommend_similar_tracks("Blinding Lights", n=5)
    """
    pass
```

### Frontend Code Style

#### **Streamlit Best Practices**
```python
# Usar caching apropiadamente
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

# Organizar en funciones claras
def create_sidebar():
    """Create and configure sidebar elements."""
    pass

def display_recommendations(recommendations):
    """Display recommendations in a user-friendly format."""
    pass

# Usar containers para layout
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Tracks", "4,600")
```

## ⚙️ Configuración del Entorno

### Setup Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/Edwin1719/spotify-recommender-pro.git
cd spotify-recommender-pro

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# 5. Configurar pre-commit hooks
pre-commit install
```

### Requirements para Desarrollo

Crea `requirements-dev.txt`:
```
# Testing
pytest==7.4.0
pytest-cov==4.1.0

# Code Quality
black==23.3.0
flake8==6.0.0
isort==5.12.0

# Pre-commit
pre-commit==3.3.3

# Documentation
sphinx==7.1.0
```

## 🧪 Testing

### Ejecutar Tests

```bash
# Ejecutar todos los tests
pytest

# Ejecutar con coverage
pytest --cov=src --cov-report=html

# Ejecutar tests específicos
pytest tests/test_recommender.py::test_similarity_calculation
```

### Escribir Tests

```python
import pytest
import pandas as pd
from src.recommender_optimized import SpotifyRecommenderOptimized

class TestSpotifyRecommender:
    """Test suite for SpotifyRecommenderOptimized."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'Track': ['Song A', 'Song B', 'Song C'],
            'Artist': ['Artist 1', 'Artist 2', 'Artist 3'],
            'Spotify Popularity': [80, 90, 70]
        })
    
    def test_initialization(self, sample_data):
        """Test recommender initialization."""
        recommender = SpotifyRecommenderOptimized(data=sample_data)
        assert recommender is not None
        assert len(recommender.data) == 3
    
    def test_recommend_similar_tracks(self, sample_data):
        """Test similarity recommendation functionality."""
        recommender = SpotifyRecommenderOptimized(data=sample_data)
        recommendations = recommender.recommend_similar_tracks("Song A", n=2)
        
        assert isinstance(recommendations, pd.DataFrame)
        assert len(recommendations) <= 2
        assert 'Track' in recommendations.columns
```

## 📝 Documentation

### Actualizar Documentación

1. **README.md**: Para cambios en instalación o uso básico
2. **Docstrings**: Para cambios en API de funciones
3. **CHANGELOG.md**: Para todos los cambios significativos

### Generar Documentación

```bash
# Generar documentación con Sphinx
cd docs
make html

# Ver documentación generada
open _build/html/index.html
```

## 🔄 Release Process

### Versionado

Seguimos [Semantic Versioning](https://semver.org/):
- **MAJOR**: Cambios incompatibles en la API
- **MINOR**: Funcionalidad nueva compatible hacia atrás
- **PATCH**: Bug fixes compatibles hacia atrás

Ejemplo: `v1.2.3`

### Changelog

Mantener `CHANGELOG.md` actualizado:

```markdown
## [1.2.0] - 2024-01-15

### Added
- Nueva funcionalidad de análisis de sentimientos
- Integración con Spotify Web API

### Changed
- Mejorado algoritmo de similaridad
- Actualizada interfaz de usuario

### Fixed
- Corregido error en cálculo de popularidad
- Arreglado problema de encoding en Windows

### Deprecated
- Función `old_recommendation_method` será removida en v2.0

### Removed
- Soporte para Python 3.7

### Security
- Actualizada dependencia con vulnerabilidad conocida
```

## 🏆 Reconocimiento

### Contributors

Todos los contributors serán reconocidos en:
- README.md
- Página de contributors en la documentación
- Release notes

### Types of Contributions

- 💻 **Code**
- 📖 **Documentation** 
- 🐛 **Bug Reports**
- 💡 **Ideas & Feature Requests**
- 🧪 **Testing**
- 🌍 **Translation**
- 🎨 **Design**

## 📞 Contacto

### Maneras de Comunicarse

- **GitHub Issues**: Para bugs y feature requests
- **GitHub Discussions**: Para preguntas y discusiones generales
- **Email**: egqa1975@gmail.com para consultas directas
- **LinkedIn**: [Edwin Quintero](https://www.linkedin.com/in/edwinquintero0329/)

### Response Times

- **Issues críticos**: 24-48 horas
- **Feature requests**: 1-2 semanas
- **Pull requests**: 3-5 días laborales
- **Questions**: 2-3 días

## 🙏 Agradecimientos

¡Gracias por contribuir al proyecto! Cada contribución, sin importar el tamaño, es valiosa y apreciada.

### ¿Primera vez contribuyendo a Open Source?

Revisa estos recursos útiles:
- [First Contributions](https://github.com/firstcontributions/first-contributions)
- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [About Pull Requests](https://help.github.com/articles/about-pull-requests/)

---

<div align="center">

**¡Happy Coding! 🚀**

[⬆ Volver al inicio](#-contributing-to-spotify-recommender-pro)

</div>