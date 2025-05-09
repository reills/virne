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
  id 1946
  arrival_time 42792.42982479114
  lifetime 1306.8950072289854
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 17
    gpu 23
    rom 43
  ]
  node [
    id 1
    label "1"
    cpu 34
    gpu 8
    rom 45
  ]
  node [
    id 2
    label "2"
    cpu 19
    gpu 30
    rom 31
  ]
  node [
    id 3
    label "3"
    cpu 23
    gpu 29
    rom 24
  ]
  node [
    id 4
    label "4"
    cpu 37
    gpu 13
    rom 37
  ]
  node [
    id 5
    label "5"
    cpu 0
    gpu 3
    rom 42
  ]
  node [
    id 6
    label "6"
    cpu 26
    gpu 50
    rom 50
  ]
  node [
    id 7
    label "7"
    cpu 27
    gpu 45
    rom 41
  ]
  node [
    id 8
    label "8"
    cpu 4
    gpu 9
    rom 35
  ]
  node [
    id 9
    label "9"
    cpu 2
    gpu 44
    rom 30
  ]
  node [
    id 10
    label "10"
    cpu 50
    gpu 25
    rom 43
  ]
  node [
    id 11
    label "11"
    cpu 20
    gpu 8
    rom 4
  ]
  edge [
    source 0
    target 1
    bw 13
  ]
  edge [
    source 1
    target 2
    bw 24
  ]
  edge [
    source 2
    target 3
    bw 27
  ]
  edge [
    source 3
    target 4
    bw 1
  ]
  edge [
    source 4
    target 5
    bw 19
  ]
  edge [
    source 5
    target 6
    bw 13
  ]
  edge [
    source 6
    target 7
    bw 3
  ]
  edge [
    source 7
    target 8
    bw 32
  ]
  edge [
    source 8
    target 9
    bw 28
  ]
  edge [
    source 9
    target 10
    bw 40
  ]
  edge [
    source 10
    target 11
    bw 5
  ]
]
