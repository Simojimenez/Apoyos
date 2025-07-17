import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calcular_reacciones(L, cargas, posiciones):
    """
    Calcula las reacciones en los apoyos de una viga simplemente apoyada
    L: longitud de la viga
    cargas: lista de cargas puntuales
    posiciones: lista de posiciones de las cargas
    """
    # Sumatoria de momentos respecto al apoyo A (izquierdo) = 0
    # RB * L - Σ(Pi * xi) = 0
    # RB = Σ(Pi * xi) / L
    
    suma_momentos = sum(P * x for P, x in zip(cargas, posiciones))
    RB = suma_momentos / L
    
    # Sumatoria de fuerzas verticales = 0
    # RA + RB - Σ(Pi) = 0
    # RA = Σ(Pi) - RB
    
    suma_cargas = sum(cargas)
    RA = suma_cargas - RB
    
    return RA, RB

def calcular_momento_cortante(L, cargas, posiciones, RA, RB, num_puntos=1000):
    """
    Calcula el momento flector y fuerza cortante a lo largo de la viga
    """
    x = np.linspace(0, L, num_puntos)
    momento = np.zeros(num_puntos)
    cortante = np.zeros(num_puntos)
    
    for i, xi in enumerate(x):
        # Fuerza cortante
        V = RA
        for P, pos in zip(cargas, posiciones):
            if xi > pos:
                V -= P
        cortante[i] = V
        
        # Momento flector
        M = RA * xi
        for P, pos in zip(cargas, posiciones):
            if xi > pos:
                M -= P * (xi - pos)
        momento[i] = M
    
    return x, momento, cortante

def crear_diagrama_viga(L, cargas, posiciones, RA, RB):
    """
    Crea el diagrama de la viga con cargas y reacciones
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Diagrama de la viga
    ax1.plot([0, L], [0, 0], 'k-', linewidth=8, label='Viga')
    
    # Apoyos
    ax1.plot(0, 0, '^', markersize=15, color='red', label=f'Apoyo A: RA = {RA:.2f} kN')
    ax1.plot(L, 0, '^', markersize=15, color='red', label=f'Apoyo B: RB = {RB:.2f} kN')
    
    # Cargas puntuales
    for i, (P, x) in enumerate(zip(cargas, posiciones)):
        ax1.arrow(x, 0.5, 0, -0.4, head_width=0.05, head_length=0.05, 
                 fc='blue', ec='blue', linewidth=2)
        ax1.text(x, 0.7, f'P{i+1}={P:.1f} kN', ha='center', va='bottom', fontsize=10)
    
    # Reacciones
    ax1.arrow(0, -0.3, 0, 0.25, head_width=0.05, head_length=0.05, 
             fc='red', ec='red', linewidth=2)
    ax1.arrow(L, -0.3, 0, 0.25, head_width=0.05, head_length=0.05, 
             fc='red', ec='red', linewidth=2)
    
    ax1.set_ylim(-0.5, 1)
    ax1.set_xlim(-0.5, L + 0.5)
    ax1.set_xlabel('Distancia (m)')
    ax1.set_title('Diagrama de la Viga con Cargas y Reacciones')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Calcular diagramas de momento y cortante
    x, momento, cortante = calcular_momento_cortante(L, cargas, posiciones, RA, RB)
    
    # Diagrama de fuerza cortante
    ax2.plot(x, cortante, 'g-', linewidth=2, label='Fuerza Cortante')
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.fill_between(x, cortante, alpha=0.3, color='green')
    ax2.set_xlabel('Distancia (m)')
    ax2.set_ylabel('Fuerza Cortante (kN)')
    ax2.set_title('Diagrama de Fuerza Cortante')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Diagrama de momento flector
    ax3.plot(x, momento, 'b-', linewidth=2, label='Momento Flector')
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.fill_between(x, momento, alpha=0.3, color='blue')
    ax3.set_xlabel('Distancia (m)')
    ax3.set_ylabel('Momento Flector (kN·m)')
    ax3.set_title('Diagrama de Momento Flector')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    return fig

def main():
    st.title("🏗️ Calculadora de Reacciones en Viga Simplemente Apoyada")
    st.markdown("---")
    
    # Configuración de la viga
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📏 Configuración de la Viga")
        L = st.number_input("Longitud de la viga (m)", min_value=0.1, value=10.0, step=0.1)
    
    with col2:
        st.subheader("📊 Número de Cargas")
        num_cargas = st.number_input("Número de cargas puntuales", min_value=1, max_value=10, value=3)
    
    st.markdown("---")
    
    # Configuración de cargas
    st.subheader("⚡ Configuración de Cargas Puntuales")
    
    cargas = []
    posiciones = []
    
    # Crear columnas para las cargas
    cols = st.columns(min(num_cargas, 3))
    
    for i in range(num_cargas):
        col_idx = i % 3
        with cols[col_idx]:
            st.write(f"**Carga {i+1}:**")
            P = st.number_input(f"Magnitud P{i+1} (kN)", value=10.0, step=1.0, key=f"carga_{i}")
            x = st.number_input(f"Posición x{i+1} (m)", min_value=0.0, max_value=L, 
                               value=min(L * (i+1) / (num_cargas+1), L), step=0.1, key=f"pos_{i}")
            cargas.append(P)
            posiciones.append(x)
    
    # Verificar que las posiciones estén dentro de la viga
    posiciones_validas = all(0 <= pos <= L for pos in posiciones)
    
    if not posiciones_validas:
        st.error("⚠️ Todas las posiciones deben estar entre 0 y L")
        return
    
    st.markdown("---")
    
    # Calcular reacciones
    RA, RB = calcular_reacciones(L, cargas, posiciones)
    
    # Mostrar resultados
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 Reacciones en los Apoyos")
        st.success(f"**Reacción en A (RA):** {RA:.3f} kN")
        st.success(f"**Reacción en B (RB):** {RB:.3f} kN")
        
        # Verificación de equilibrio
        suma_fuerzas = RA + RB - sum(cargas)
        st.info(f"**Verificación ΣFy:** {suma_fuerzas:.6f} kN")
        
        if abs(suma_fuerzas) < 1e-10:
            st.success("✅ Equilibrio de fuerzas verificado")
        else:
            st.error("❌ Error en el equilibrio de fuerzas")
    
    with col4:
        st.subheader("📋 Resumen de Cargas")
        
        # Crear tabla con las cargas
        df_cargas = pd.DataFrame({
            'Carga': [f'P{i+1}' for i in range(num_cargas)],
            'Magnitud (kN)': cargas,
            'Posición (m)': posiciones,
            'Momento (kN·m)': [P * x for P, x in zip(cargas, posiciones)]
        })
        
        st.dataframe(df_cargas, hide_index=True)
        
        st.write(f"**Suma total de cargas:** {sum(cargas):.3f} kN")
        st.write(f"**Suma total de momentos:** {sum(P * x for P, x in zip(cargas, posiciones)):.3f} kN·m")
    
    # Opción para mostrar gráfico
    mostrar_grafico = st.checkbox("📈 Mostrar diagramas", value=True)
    
    if mostrar_grafico:
        st.markdown("---")
        st.subheader("📈 Diagramas de la Viga")
        
        fig = crear_diagrama_viga(L, cargas, posiciones, RA, RB)
        st.pyplot(fig)
        
        # Calcular valores máximos
        x, momento, cortante = calcular_momento_cortante(L, cargas, posiciones, RA, RB)
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("📊 Valores Máximos")
            momento_max = np.max(np.abs(momento))
            cortante_max = np.max(np.abs(cortante))
            
            st.metric("Momento máximo", f"{momento_max:.3f} kN·m")
            st.metric("Cortante máximo", f"{cortante_max:.3f} kN")
        
        with col6:
            st.subheader("📍 Ubicaciones Críticas")
            idx_momento_max = np.argmax(np.abs(momento))
            idx_cortante_max = np.argmax(np.abs(cortante))
            
            st.write(f"**Momento máximo en:** x = {x[idx_momento_max]:.3f} m")
            st.write(f"**Cortante máximo en:** x = {x[idx_cortante_max]:.3f} m")
    
    # Información teórica
    st.markdown("---")
    st.markdown("### 📚 Ecuaciones de Equilibrio")
    
    col7, col8 = st.columns(2)
    
    with col7:
        st.markdown("**Equilibrio de Fuerzas:**")
        st.latex(r"\sum F_y = 0")
        st.latex(r"R_A + R_B - \sum P_i = 0")
        
    with col8:
        st.markdown("**Equilibrio de Momentos (respecto a A):**")
        st.latex(r"\sum M_A = 0")
        st.latex(r"R_B \cdot L - \sum (P_i \cdot x_i) = 0")
    
    st.markdown("**Solución:**")
    st.latex(r"R_B = \frac{\sum (P_i \cdot x_i)}{L}")
    st.latex(r"R_A = \sum P_i - R_B")

if __name__ == "__main__":
    main()