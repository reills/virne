graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 549
  arrival_time 10358.250392488731
  lifetime 1817.271250285818
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 12
    gpu 39
    rom 29
  ]
  node [
    id 1
    label "1"
    cpu 44
    gpu 41
    rom 20
  ]
  node [
    id 2
    label "2"
    cpu 0
    gpu 45
    rom 17
  ]
  node [
    id 3
    label "3"
    cpu 19
    gpu 30
    rom 2
  ]
  node [
    id 4
    label "4"
    cpu 19
    gpu 12
    rom 42
  ]
  node [
    id 5
    label "5"
    cpu 46
    gpu 50
    rom 26
  ]
  edge [
    source 0
    target 1
    bw 35
  ]
  edge [
    source 1
    target 2
    bw 29
  ]
  edge [
    source 2
    target 3
    bw 27
  ]
  edge [
    source 3
    target 4
    bw 9
  ]
  edge [
    source 4
    target 5
    bw 26
  ]
]
