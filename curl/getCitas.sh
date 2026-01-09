# script to get citas, debe poder enviar parametros a la siguiente funcion
# async function getDisponibilidad(req, res) {
#   try {
#     const { empresaId, especialistaId, pacienteId, fecha } = req.query; // Para buscar por teléfono
#     if (empresaId) {
#       const dispo = await generarDisponibilidad(
#         empresaId,
#         especialistaId,
#         fecha,
#         pacienteId
#       );
#       res.json(dispo);
#     } else {
#       res.json({ data: "vacio" });
#     }
#   } catch (err) {
#     res.status(400).send(err.message);
#   }
# }/*
curl -X GET "https://us-central1-odontoplus-4db47.cloudfunctions.net/api/v1/citas?empresaId=hIntsAEzBwy8Hwi4DNcf&especialistaId=EjEoM4k4RpkJWf585ZSc&fecha=2025-11-18" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJFakVvTTRrNFJwa0pXZjU4NVpTcSIsImVtcHJlc2EiOiJoSW50c0FFekJ3eThId2k0RE5jZiIsImlhdCI6MTcyODA1Mzk5M30.HCoHtyuJYtKcNv0imD2nCAmxxoB89PL1g7UIC6MYmAo"
     

# curl -X GET https://us-central1-odontoplus-4db47.cloudfunctions.net/api/v1/citas \
#      -H "Content-Type: application/json" \
#      -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJFakVvTTRrNFJwa0pXZjU4NVpTcSIsImVtcHJlc2EiOiJoSW50c0FFekJ3eThId2k0RE5jZiIsImlhdCI6MTcyODA1Mzk5M30.HCoHtyuJYtKcNv0imD2nCAmxxoB89PL1g7UIC6MYmAo"
#      -d '{
#             "empresa_id": "A0OZsgiMQQVtwxMhN6Um",
#             "especialista_id": "GMcKghlgHvTkoPxj9t4X",
#             "fecha_inicio": "2025-11-10T00:00:00Z",
#             "fecha_fin": "2025-11-30T23:59:59Z"
#             }'
   